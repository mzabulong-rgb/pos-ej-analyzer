import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

from flask import Flask, jsonify, request
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pos-ej-analyzer")


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    "",
).strip()

GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-3.8-flash",
).strip()

GEMINI_BATCH_MAX_CHARS = int(
    os.environ.get(
        "GEMINI_BATCH_MAX_CHARS",
        "45000",
    )
)

if not GEMINI_API_KEY:
    logger.warning(
        "GEMINI_API_KEY is not configured."
    )


# ============================================================
# DOCUMENT TYPES
# ============================================================

PRODUCTION_TYPES = {
    "BAR",
    "WINDOW",
    "NACHO",
    "KITCHEN",
    "MEAT",
    "MIDDLE",
    "PANTRY",
    "GRILL",
    "FRYER",
    "DESSERT",
    "BEVERAGE",
    "SERVICE BAR",
}

CUSTOMER_TYPES = {
    "BILL",
    "PRINT BILL",
    "CUSTOMER BILL",
    "REBILL",
    "RE-BILL",
    "REPRINT",
    "BILL REPRINT",
    "INVOICE",
    "OFFICIAL RECEIPT",
    "SALES INVOICE",
    "BILL CANCELLATION",
    "BILL CANCEL",
    "CANCELLATION",
    "RETURN",
    "REFUND",
    "CREDIT MEMO",
}

KNOWN_TYPES = (
    PRODUCTION_TYPES
    | CUSTOMER_TYPES
)


# ============================================================
# OUTPUT SCHEMA
# ============================================================

class POSException(BaseModel):
    slip_number: str = ""
    transaction_number: str = ""
    bill_slip_number: str = ""
    invoice_number: str = ""
    return_slip_number: str = ""
    pos_id: str = ""
    staff: str = ""
    business_date: str = ""

    initial_bill_amount: Optional[float] = None
    rebilled_amount: Optional[float] = None
    final_invoice_amount: Optional[float] = None
    amount_difference: Optional[float] = None

    number_of_bill_prints: int = 0
    bill_amount_trail: str = ""

    items_removed: str = ""
    items_added: str = ""

    initial_discount: str = ""
    final_discount: str = ""
    discount_changes: str = ""
    number_of_discount_changes: int = 0

    initial_tender: str = ""
    final_tender: str = ""
    tender_changes: str = ""
    number_of_tender_changes: int = 0

    flag_categories: List[str] = Field(
        default_factory=list
    )

    risk_tier: str = ""
    finding: str = ""
    evidence_trail: str = ""
    rule_triggered: str = ""


class EJAnalysis(BaseModel):
    exceptions: List[POSException] = Field(
        default_factory=list
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_gemini_client():
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is missing."
        )

    return genai.Client(
        api_key=GEMINI_API_KEY
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(
    text: str,
) -> str:
    text = (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\x00", "")
    )

    return "\n".join(
        line.rstrip()
        for line in text.split("\n")
    )


def normalize_title(
    line: str,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        line.strip().upper(),
    )


# ============================================================
# SLIP EXTRACTION
# ============================================================

def extract_slip_numbers(
    text: str,
) -> List[str]:
    patterns = [
        r"(?im)^\s*Slip\s*:\s*([A-Z0-9\-]+)",
        r"(?im)^\s*Slip\s+Number\s*:\s*([A-Z0-9\-]+)",
        r"(?im)^\s*Slip\s*No\.?\s*:\s*([A-Z0-9\-]+)",
    ]

    found: List[str] = []

    for pattern in patterns:
        for value in re.findall(
            pattern,
            text,
        ):
            value = value.strip()

            if (
                value
                and value not in found
            ):
                found.append(
                    value
                )

    return found


# ============================================================
# DOCUMENT PARSING
# ============================================================

def is_document_heading(
    line: str,
) -> bool:
    return (
        normalize_title(line)
        in KNOWN_TYPES
    )


def classify_document(
    block: str,
) -> str:
    for line in block.splitlines()[:25]:
        title = normalize_title(
            line
        )

        if title in KNOWN_TYPES:
            return title

    return "UNKNOWN"


def split_into_document_blocks(
    ej_content: str,
) -> List[str]:
    blocks: List[str] = []
    current: List[str] = []

    for line in ej_content.splitlines():
        if (
            is_document_heading(line)
            and current
        ):
            block = "\n".join(
                current
            ).strip()

            if block:
                blocks.append(
                    block
                )

            current = [line]

        else:
            current.append(
                line
            )

    if current:
        block = "\n".join(
            current
        ).strip()

        if block:
            blocks.append(
                block
            )

    return blocks


def group_blocks_by_slip(
    blocks: List[str],
) -> Dict[str, List[str]]:
    grouped: Dict[
        str,
        List[str],
    ] = {}

    for block in blocks:
        slip_numbers = extract_slip_numbers(
            block
        )

        for slip_number in slip_numbers:
            grouped.setdefault(
                slip_number,
                [],
            ).append(
                block
            )

    return grouped


# ============================================================
# FALLBACK SLIP CONTEXT
# ============================================================

def add_missing_slip_contexts(
    ej_content: str,
    grouped: Dict[str, List[str]],
) -> None:
    pattern = re.compile(
        r"(?im)^\s*Slip\s*:\s*([A-Z0-9\-]+)"
    )

    matches = list(
        pattern.finditer(
            ej_content
        )
    )

    for index, match in enumerate(
        matches
    ):
        slip_number = (
            match.group(1).strip()
        )

        if slip_number in grouped:
            continue

        start_position = max(
            0,
            match.start() - 1200,
        )

        if index + 1 < len(matches):
            end_position = (
                matches[
                    index + 1
                ].start()
            )
        else:
            end_position = min(
                len(ej_content),
                match.end() + 5000,
            )

        context = ej_content[
            start_position:end_position
        ].strip()

        if context:
            grouped.setdefault(
                slip_number,
                [],
            ).append(
                context
            )


# ============================================================
# BOILERPLATE REMOVAL
# ============================================================

def remove_boilerplate(
    text: str,
) -> str:
    phrases = [
        "THIS DOCUMENT IS NOT",
        "VALID FOR CLAIM OF INPUT TAX",
        "BISTRO AMERICANO CORP.",
        "G/F UNIT",
        "LIFESTYLE MALL",
        "BRGY.",
        "VAT REGISTERED TIN",
        "SERIAL NO:",
        "MIN:",
        "POS PROVIDER:",
        "PERFORMANCE ADVANTAGE SYSTEMS",
        "DATE OF ISSUANCE:",
        "EFFECTIVITY DATE:",
        "ACCRED. NO.:",
        "PTU NO.:",
    ]

    kept: List[str] = []

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        if set(stripped) <= {
            "-",
            "=",
            "_",
            " ",
        }:
            continue

        upper_line = (
            stripped.upper()
        )

        if any(
            phrase in upper_line
            for phrase in phrases
        ):
            continue

        kept.append(
            stripped
        )

    return "\n".join(
        kept
    )


# ============================================================
# BUILD COMPLETE SLIP TRAIL
# ============================================================

def build_slip_trail(
    slip_number: str,
    blocks: List[str],
) -> Tuple[str, List[str]]:
    compacted: List[str] = []
    document_types: List[str] = []
    seen = set()

    for block in blocks:
        cleaned = remove_boilerplate(
            block
        ).strip()

        if not cleaned:
            continue

        fingerprint = cleaned[:600]

        if fingerprint in seen:
            continue

        seen.add(
            fingerprint
        )

        document_type = classify_document(
            block
        )

        document_types.append(
            document_type
        )

        compacted.append(
            (
                f"[DOCUMENT TYPE: {document_type}]\n"
                f"{cleaned}"
            )
        )

    trail = (
        f"SLIP NUMBER: {slip_number}\n\n"
        + "\n\n".join(
            compacted
        )
    )

    return (
        trail,
        document_types,
    )


# ============================================================
# FIRST-STAGE SCREENING
# ============================================================

def is_production_only(
    document_types: List[str],
) -> bool:
    known = [
        document_type
        for document_type in document_types
        if document_type != "UNKNOWN"
    ]

    return (
        bool(known)
        and all(
            document_type in PRODUCTION_TYPES
            for document_type in known
        )
    )


def should_review(
    trail: str,
    document_types: List[str],
) -> Tuple[bool, str]:
    if is_production_only(
        document_types
    ):
        return (
            False,
            "Production-printer records only.",
        )

    high_priority = {
        "REBILL",
        "RE-BILL",
        "REPRINT",
        "BILL REPRINT",
        "BILL CANCELLATION",
        "BILL CANCEL",
        "CANCELLATION",
        "RETURN",
        "REFUND",
        "CREDIT MEMO",
    }

    if any(
        document_type in high_priority
        for document_type in document_types
    ):
        return (
            True,
            "High-priority customer document found.",
        )

    customer_count = sum(
        1
        for document_type in document_types
        if document_type in CUSTOMER_TYPES
    )

    if customer_count >= 2:
        return (
            True,
            "Multiple customer-facing documents found.",
        )

    keywords = [
        "CANCEL",
        "VOID",
        "DELETE",
        "REMOVE",
        "RETURN",
        "REFUND",
        "REVERSAL",
        "DISCOUNT",
        "SENIOR",
        "PWD",
        "VAT EXEMPT",
        "TENDER",
        "PAYMENT",
        "PAYMAYA",
        "GCASH",
        "CREDIT CARD",
        "DEBIT CARD",
        "CASH",
    ]

    upper_trail = trail.upper()

    has_keyword = any(
        keyword in upper_trail
        for keyword in keywords
    )

    if (
        customer_count >= 1
        and has_keyword
    ):
        return (
            True,
            "Customer document with review signal found.",
        )

    if "INVOICE" in document_types:
        return (
            True,
            "Invoice and settlement trail found.",
        )

    if has_keyword:
        return (
            True,
            "Potential transaction review signal found.",
        )

    return (
        False,
        "No configured transaction-review signal found.",
    )


# ============================================================
# BATCHING
# ============================================================

def create_batches(
    candidate_trails: List[
        Tuple[str, str]
    ],
) -> List[
    List[
        Tuple[str, str]
    ]
]:
    batches: List[
        List[
            Tuple[str, str]
        ]
    ] = []

    current: List[
        Tuple[str, str]
    ] = []

    current_chars = 0

    for (
        slip_number,
        trail,
    ) in candidate_trails:
        trail_size = len(
            trail
        )

        if (
            current
            and current_chars + trail_size
            > GEMINI_BATCH_MAX_CHARS
        ):
            batches.append(
                current
            )

            current = []
            current_chars = 0

        current.append(
            (
                slip_number,
                trail,
            )
        )

        current_chars += (
            trail_size
        )

    if current:
        batches.append(
            current
        )

    return batches


# ============================================================
# GEMINI SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a POS Electronic Journal transaction exception analyst.

You will receive complete transaction trails grouped by Slip Number.

Evaluate every supplied Slip independently.

A flagged transaction is a POTENTIAL IRREGULARITY ONLY.

Never state or imply that fraud, theft, misconduct, or financial loss
definitely occurred.

============================================================
DOCUMENT CLASSIFICATION
============================================================

BAR, WINDOW, NACHO, KITCHEN, MEAT, MIDDLE, PANTRY, GRILL,
FRYER, DESSERT, BEVERAGE, and SERVICE BAR are production-printer
documents.

Production-printer documents are NOT customer Print Bills.

Repeated production documents never constitute Multiple Print Bill.

Customer-facing documents include actual BILL, PRINT BILL,
CUSTOMER BILL, REBILL, BILL REPRINT, INVOICE,
BILL CANCELLATION, RETURN, and REFUND.

Do not count a final INVOICE as a customer Print Bill unless the
document itself is explicitly identified as a Print Bill.

============================================================
EXCEPTION PATTERNS
============================================================

Consider patterns including:

1. Multiple customer Print Bills where the amount materially changes.
2. Customer Print Bill followed by item removal.
3. Material item addition or substitution after a customer Print Bill.
4. Bill cancellation followed by replacement or rebill.
5. Repeated bill cancellations.
6. Material amount decrease between earlier and later customer bills.
7. Discount added, removed, replaced, or materially changed after a
   customer Print Bill.
8. Multiple or unusual discount combinations requiring review.
9. VAT-exempt treatment combined with another discount where the
   sequence warrants review.
10. Tender or payment reversal/change associated with another unusual
    transaction event.
11. Return or refund with an unusual preceding or subsequent trail.
12. Material transaction changes after apparent settlement.
13. Reprints associated with material transaction changes.
14. Other materially inconsistent transaction sequences that warrant
    human investigation.

============================================================
DO NOT FLAG SOLELY BECAUSE
============================================================

Do not flag solely because:

- the same Slip Number appears repeatedly;
- production-printer records repeat;
- BAR, WINDOW, NACHO, KITCHEN, MEAT, or MIDDLE prints repeat;
- an invoice exists;
- a transaction crosses midnight;
- a discount exists;
- multiple tender exists;
- an unchanged customer bill was reprinted.

============================================================
AMOUNTS
============================================================

Initial Bill Amount:
The first identifiable CUSTOMER Print Bill amount.

Rebilled Amount:
The last identifiable CUSTOMER Print Bill amount before final invoice
or settlement.

Final Invoice Amount:
The amount finally invoiced or settled.

Amount Difference:
Rebilled Amount minus Initial Bill Amount.

Number of Bill Prints:
The number of CUSTOMER Print Bills only.

============================================================
EVIDENCE
============================================================

Every exception must be supported by specific evidence from the
provided Slip trail.

Explain the relevant sequence chronologically in evidence_trail.

Do not invent Slip Numbers, transaction numbers, amounts, timestamps,
items, discounts, documents, tenders, invoices, cancellations,
returns, or refunds.

If a value is unavailable, return an empty string or null.

============================================================
RISK TIER
============================================================

Use exactly one:

High
Medium
Low

High:
Strong multi-event exception or high-impact control concern.

Medium:
Meaningful anomaly requiring investigation without enough evidence
to classify High.

Low:
Weak but identifiable anomaly worth monitoring.

Return only Slips that warrant human review.

If none warrant review, return an empty exceptions array.
"""


# ============================================================
# GEMINI BATCH ANALYSIS
# ============================================================

def analyze_batch(
    client,
    store: str,
    ej_file_name: str,
    batch_number: int,
    batch: List[
        Tuple[str, str]
    ],
    approved_rules: List,
) -> Tuple[EJAnalysis, float]:
    slip_numbers = [
        slip_number
        for (
            slip_number,
            _,
        ) in batch
    ]

    trails = "\n\n".join(
        trail
        for (
            _,
            trail,
        ) in batch
    )

    user_prompt = f"""
STORE:
{store}

SOURCE EJ FILE:
{ej_file_name}

BATCH NUMBER:
{batch_number}

SLIP NUMBERS:
{json.dumps(
    slip_numbers,
    ensure_ascii=False
)}

APPROVED LEGITIMATE-PATTERN RULES:
{json.dumps(
    approved_rules,
    ensure_ascii=False
)}

Approved rules may suppress a matching pattern only when every rule
condition is satisfied.

An approved rule must not suppress a separate material anomaly in the
same Slip.

TRANSACTION TRAILS START
============================================================
{trails}
============================================================
TRANSACTION TRAILS END

Analyze every supplied Slip.

Return only exceptions requiring human review.
"""

    started = time.time()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,
            response_mime_type=(
                "application/json"
            ),
            response_schema=EJAnalysis,
        ),
    )

    elapsed = round(
        time.time() - started,
        3,
    )

    if not response.text:
        raise RuntimeError(
            (
                "Gemini returned an empty "
                f"response for batch {batch_number}."
            )
        )

    try:
        result = EJAnalysis.model_validate_json(
            response.text
        )

    except Exception as exc:
        raise RuntimeError(
            (
                "Invalid Gemini response "
                f"for batch {batch_number}: {exc}"
            )
        ) from exc

    return (
        result,
        elapsed,
    )


# ============================================================
# DEDUPLICATE EXCEPTIONS
# ============================================================

def deduplicate_exceptions(
    exceptions: List[POSException],
) -> List[POSException]:
    unique: Dict[
        Tuple[str, str, str],
        POSException,
    ] = {}

    for exception in exceptions:
        key = (
            exception.slip_number.strip(),
            exception.rule_triggered.strip(),
            exception.finding.strip(),
        )

        if key not in unique:
            unique[key] = exception

    return list(
        unique.values()
    )

# ============================================================
# COMPLETE EJ ANALYSIS
# ============================================================

def analyze_ej(
    store: str,
    ej_file_name: str,
    ej_content: str,
    approved_rules=None,
) -> Tuple[
    List[POSException],
    Dict,
]:
    if approved_rules is None:
        approved_rules = []

    normalized = normalize_text(
        ej_content
    )

    blocks = split_into_document_blocks(
        normalized
    )

    grouped = group_blocks_by_slip(
        blocks
    )

    add_missing_slip_contexts(
        normalized,
        grouped,
    )

    candidates: List[
        Tuple[str, str]
    ] = []

    skipped = 0

    for (
        slip_number,
        slip_blocks,
    ) in grouped.items():
        (
            trail,
            document_types,
        ) = build_slip_trail(
            slip_number,
            slip_blocks,
        )

        (
            review,
            _,
        ) = should_review(
            trail,
            document_types,
        )

        if review:
            candidates.append(
                (
                    slip_number,
                    trail,
                )
            )

        else:
            skipped += 1

    batches = create_batches(
        candidates
    )

    all_exceptions: List[
        POSException
    ] = []

    batch_seconds: List[
        float
    ] = []

    if batches:
        client = get_gemini_client()

        for (
            batch_number,
            batch,
        ) in enumerate(
            batches,
            start=1,
        ):
            (
                result,
                elapsed,
            ) = analyze_batch(
                client=client,
                store=store,
                ej_file_name=ej_file_name,
                batch_number=batch_number,
                batch=batch,
                approved_rules=approved_rules,
            )

            all_exceptions.extend(
                result.exceptions
            )

            batch_seconds.append(
                elapsed
            )

    all_exceptions = (
        deduplicate_exceptions(
            all_exceptions
        )
    )

    analysis_summary = {
        "raw_content_characters": (
            len(normalized)
        ),
        "document_blocks_found": (
            len(blocks)
        ),
        "slips_found": (
            len(grouped)
        ),
        "candidate_slips": (
            len(candidates)
        ),
        "production_or_low_signal_slips_skipped": (
            skipped
        ),
        "gemini_batches_processed": (
            len(batches)
        ),
        "batch_processing_seconds": (
            batch_seconds
        ),
        "processing_complete": True,
    }

    return (
        all_exceptions,
        analysis_summary,
    )


# ============================================================
# HEALTH ENDPOINTS
# ============================================================

@app.get("/")
def health():
    return jsonify(
        {
            "success": True,
            "service": "pos-ej-analyzer",
            "status": "healthy",
            "model": GEMINI_MODEL,
            "batch_max_characters": (
                GEMINI_BATCH_MAX_CHARS
            ),
        }
    )


@app.get("/health")
def health_explicit():
    return jsonify(
        {
            "success": True,
            "service": "pos-ej-analyzer",
            "status": "healthy",
            "model": GEMINI_MODEL,
        }
    )


# ============================================================
# ANALYZE ENDPOINT
#
# Both of these POST routes are supported:
# POST /
# POST /analyze
# ============================================================

@app.post("/")
@app.post("/analyze")
def analyze():
    request_started = time.time()

    try:
        payload = request.get_json(
            silent=True
        )

        if not isinstance(
            payload,
            dict,
        ):
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Request body must be valid JSON."
                    ),
                }
            ), 400

        mode = str(
            payload.get(
                "mode",
                "analyze_ej",
            )
        ).strip()

        store = str(
            payload.get(
                "store",
                "",
            )
        ).strip()

        ej_file_name = str(
            payload.get(
                "ej_file_name",
                "",
            )
        ).strip()

        ej_content = payload.get(
            "ej_content",
            "",
        )

        approved_rules = payload.get(
            "approved_rules",
            [],
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if mode != "analyze_ej":
            return jsonify(
                {
                    "success": False,
                    "error": (
                        f"Unsupported mode: {mode}"
                    ),
                }
            ), 400

        if not store:
            return jsonify(
                {
                    "success": False,
                    "error": "store is required.",
                }
            ), 400

        if not ej_file_name:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "ej_file_name is required."
                    ),
                }
            ), 400

        if not isinstance(
            ej_content,
            str,
        ):
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "ej_content must be text."
                    ),
                }
            ), 400

        if not ej_content.strip():
            return jsonify(
                {
                    "success": False,
                    "error": "ej_content is empty.",
                }
            ), 400

        if not isinstance(
            approved_rules,
            list,
        ):
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "approved_rules must be an array."
                    ),
                }
            ), 400

        # ----------------------------------------------------
        # ANALYSIS
        # ----------------------------------------------------

        (
            exceptions,
            analysis_summary,
        ) = analyze_ej(
            store=store,
            ej_file_name=ej_file_name,
            ej_content=ej_content,
            approved_rules=approved_rules,
        )

        processing_seconds = round(
            time.time() - request_started,
            3,
        )

        # ----------------------------------------------------
        # RESPONSE
        #
        # Existing fields are preserved for Power Automate,
        # Parse JSON, and the POS Exceptions list.
        # ----------------------------------------------------

        return jsonify(
            {
                "success": True,
                "mode": "analyze_ej",
                "model": GEMINI_MODEL,
                "store": store,
                "ej_file_name": ej_file_name,
                "exception_count": len(
                    exceptions
                ),
                "exceptions": [
                    item.model_dump()
                    for item in exceptions
                ],
                "processing_seconds": (
                    processing_seconds
                ),
                "analysis_summary": (
                    analysis_summary
                ),
            }
        ), 200

    except Exception as exc:
        logger.exception(
            "EJ analysis failed."
        )

        return jsonify(
            {
                "success": False,
                "mode": "analyze_ej",
                "error": str(exc),
            }
        ), 500


# ============================================================
# LOCAL ENTRY POINT
# ============================================================

if __name__ == "__main__":
    port = int(
        os.environ.get(
            "PORT",
            "8080",
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
    )
