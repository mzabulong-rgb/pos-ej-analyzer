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

# Maximum compacted characters included in one Gemini request.
# This can later be changed through Cloud Run environment variables.
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

PRODUCTION_DOCUMENT_TYPES = {
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

CUSTOMER_DOCUMENT_TYPES = {
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

KNOWN_DOCUMENT_TYPES = (
    PRODUCTION_DOCUMENT_TYPES
    | CUSTOMER_DOCUMENT_TYPES
)


# ============================================================
# OUTPUT SCHEMAS
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

def normalize_ej_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\x00", "")

    lines = [
        line.rstrip()
        for line in text.split("\n")
    ]

    return "\n".join(lines)


def normalize_title(line: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        line.strip().upper(),
    )


def extract_slip_numbers(text: str) -> List[str]:
    patterns = [
        r"(?im)^\s*Slip\s*:\s*([A-Z0-9\-]+)",
        r"(?im)^\s*Slip\s+Number\s*:\s*([A-Z0-9\-]+)",
        r"(?im)^\s*Slip\s*No\.?\s*:\s*([A-Z0-9\-]+)",
    ]

    slip_numbers: List[str] = []

    for pattern in patterns:
        matches = re.findall(
            pattern,
            text,
        )

        for match in matches:
            slip_number = match.strip()

            if (
                slip_number
                and slip_number not in slip_numbers
            ):
                slip_numbers.append(
                    slip_number
                )

    return slip_numbers


# ============================================================
# DOCUMENT PARSING
# ============================================================

def is_document_heading(line: str) -> bool:
    return normalize_title(line) in KNOWN_DOCUMENT_TYPES


def classify_document(block: str) -> str:
    lines = block.splitlines()

    for line in lines[:25]:
        title = normalize_title(line)

        if title in KNOWN_DOCUMENT_TYPES:
            return title

    return "UNKNOWN"


def split_into_document_blocks(
    ej_content: str,
) -> List[str]:
    """
    Splits the EJ whenever a recognized document heading begins.

    This keeps production records and customer-facing documents as
    individual blocks.
    """

    lines = ej_content.splitlines()

    blocks: List[str] = []
    current_block: List[str] = []

    for line in lines:
        if is_document_heading(line):
            if current_block:
                completed_block = "\n".join(
                    current_block
                ).strip()

                if completed_block:
                    blocks.append(
                        completed_block
                    )

            current_block = [line]
        else:
            current_block.append(line)

    if current_block:
        completed_block = "\n".join(
            current_block
        ).strip()

        if completed_block:
            blocks.append(
                completed_block
            )

    return blocks


def group_blocks_by_slip(
    blocks: List[str],
) -> Dict[str, List[str]]:
    grouped_slips: Dict[str, List[str]] = {}

    for block in blocks:
        slip_numbers = extract_slip_numbers(
            block
        )

        for slip_number in slip_numbers:
            grouped_slips.setdefault(
                slip_number,
                [],
            ).append(block)

    return grouped_slips


def add_missing_slip_contexts(
    ej_content: str,
    grouped_slips: Dict[str, List[str]],
) -> None:
    """
    Fallback for records that contain a Slip Number but were not
    captured under a recognized heading.
    """

    slip_pattern = re.compile(
        r"(?im)^\s*Slip\s*:\s*([A-Z0-9\-]+)"
    )

    matches = list(
        slip_pattern.finditer(
            ej_content
        )
    )

    for index, match in enumerate(matches):
        slip_number = match.group(1).strip()

        if slip_number in grouped_slips:
            continue

        start_position = max(
            0,
            match.start() - 1200,
        )

        if index + 1 < len(matches):
            end_position = matches[
                index + 1
            ].start()
        else:
            end_position = min(
                len(ej_content),
                match.end() + 5000,
            )

        context = ej_content[
            start_position:end_position
        ].strip()

        if context:
            grouped_slips.setdefault(
                slip_number,
                [],
            ).append(context)


# ============================================================
# TRAIL COMPACTION
# ============================================================

def remove_boilerplate(text: str) -> str:
    boilerplate_phrases = [
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

    compacted_lines: List[str] = []

    for line in text.splitlines():
        stripped_line = line.strip()

        if not stripped_line:
            continue

        if set(stripped_line) <= {
            "-",
            "=",
            "_",
            " ",
        }:
            continue

        upper_line = stripped_line.upper()

        if any(
            phrase in upper_line
            for phrase in boilerplate_phrases
        ):
            continue

        compacted_lines.append(
            stripped_line
        )

    return "\n".join(
        compacted_lines
    )


def build_slip_trail(
    slip_number: str,
    blocks: List[str],
) -> Tuple[str, List[str]]:
    compacted_blocks: List[str] = []
    document_types: List[str] = []
    seen_blocks = set()

    for block in blocks:
        document_type = classify_document(
            block
        )

        compacted_block = remove_boilerplate(
            block
        ).strip()

        if not compacted_block:
            continue

        # Prevent identical blocks from appearing twice.
        fingerprint = compacted_block[:600]

        if fingerprint in seen_blocks:
            continue

        seen_blocks.add(
            fingerprint
        )

        document_types.append(
            document_type
        )

        compacted_blocks.append(
            (
                f"[DOCUMENT TYPE: {document_type}]\n"
                f"{compacted_block}"
            )
        )

    trail = (
        f"SLIP NUMBER: {slip_number}\n\n"
        + "\n\n".join(compacted_blocks)
    )

    return trail, document_types


# ============================================================
# DETERMINISTIC SCREENING
# ============================================================

def is_production_only(
    document_types: List[str],
) -> bool:
    known_types = [
        document_type
        for document_type in document_types
        if document_type != "UNKNOWN"
    ]

    if not known_types:
        return False

    return all(
        document_type in PRODUCTION_DOCUMENT_TYPES
        for document_type in known_types
    )


def count_customer_documents(
    document_types: List[str],
) -> int:
    return sum(
        1
        for document_type in document_types
        if document_type in CUSTOMER_DOCUMENT_TYPES
    )


def should_review_with_gemini(
    trail: str,
    document_types: List[str],
) -> Tuple[bool, str]:
    """
    Python only performs first-stage screening.

    Gemini still decides whether a transaction should be returned as
    an exception.
    """

    if is_production_only(document_types):
        return (
            False,
            "Production-printer records only.",
        )

    customer_document_t_customer_documents(
            document_types
        )
    )

    high_priority_documents = {
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
        document_type in high_priority_documents
        for document_type in document_types
    ):
        return (
            True,
            "High-priority transaction document found.",
        )

    if customer_document_count >= 2:
        return (
            True,
            "Multiple customer-facing documents found.",
        )

    upper_trail = trail.upper()

    review_keywords = [
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

    has_review_keyword = any(
        keyword in upper_trail
        for keyword in review_keywords
    )

    if (
        customer_document_count >= 1
        and has_review_keyword
    ):
        return (
            True,
            "Customer document with a review signal found.",
        )

    if "INVOICE" in document_types:
        return (
            True,
            "Invoice and settlement trail found.",
        )

    if has_review_keyword:
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
    candidate_trails: List[Tuple[str, str]],
) -> List[List[Tuple[str, str]]]:
    batches: List[List[Tuple[str, str]]] = []

    current_batch: List[Tuple[str, str]] = []
    current_character_count = 0

    for slip_number, trail in candidate_trails:
        trail_character_count = len(trail)

        if (
            current_batch
            and (
                current_character_count
                + trail_character_count
                > GEMINI_BATCH_MAX_CHARS
            )
        ):
            batches.append(
                current_batch
            )

            current_batch = []
            current_character_count = 0

        current_batch.append(
            (
                slip_number,
                trail,
            )
        )

        current_character_count += (
            trail_character_count
        )

    if current_batch:
        batches.append(
            current_batch
        )

    return batches


# ============================================================
# GEMINI PROMPT
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
FRYER, DESSERT, BEVERAGE and SERVICE BAR are production-printer
documents.

Production-printer documents are NOT customer Print Bills.

Repeated production documents never constitute Multiple Print Bill.

Customer-facing documents include actual BILL, PRINT BILL,
CUSTOMER BILL, REBILL, BILL REPRINT, INVOICE,
BILL CANCELLATION, RETURN and REFUND documents.

Do not count a final INVOICE as a customer Print Bill unless the
document is explicitly identified as a Print Bill.

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
- BAR, WINDOW, NACHO, KITCHEN, MEAT or MIDDLE prints repeat;
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

Explain the sequence chronologically in evidence_trail.

Do not invent Slip Numbers, transaction numbers, amounts, timestamps,
items, discounts, documents, tenders, invoices, cancellations,
returns, or refunds.

If a value cannot be identified, return an empty string or null.

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
Meaningful anomaly requiring investigation, without enough evidence
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
    batch: List[Tuple[str, str]],
    approved_rules: List,
) -> Tuple[EJAnalysis, float]:
    slip_numbers = [
        slip_number
        for slip_number, _ in batch
    ]

    combined_trails = "\n\n".join(
        trail
        for _, trail in batch
    )

    user_prompt = f"""
STORE:
{store}

SOURCE EJ FILE:
{ej_file_name}

BATCH NUMBER:
{batch_number}

SLIP NUMBERS:
{json.dumps(slip_numbers, ensure_ascii=False)}

APPROVED LEGITIMATE-PATTERN RULES:
{json.dumps(approved_rules, ensure_ascii=False)}

Approved rules may suppress a matching legitimate pattern only when
all rule conditions are satisfied.

An approved rule must not suppress a separate material anomaly in the
same Slip.

TRANSACTION TRAILS START
============================================================
{combined_trails}
============================================================
TRANSACTION TRAILS END

Analyze every supplied Slip.

Return only exceptions that warrant human review.
"""

    started = time.time()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,
            response_mime_type="application/json",
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
                "Gemini returned an empty response "
                f"for batch {batch_number}."
            )
        )

    try:
        result = EJAnalysis.model_validate_json(
            response.text
        )
    except Exception as exc:
        logger.exception(
            "Gemini response failed validation."
        )

        raise RuntimeError(
            (
                "Invalid structured Gemini response "
                f"for batch {batch_number}: {exc}"
            )
        ) from exc

    return result, elapsed


def deduplicate_exceptions(
    exceptions: List[POSException],
) -> Listdeduplicated: Dict[
        Tuple[str, str, str],
        POSException,
    ] = {}

    for exception in exceptions:
        key = (
            exception.slip_number.strip(),
            exception.rule_triggered.strip(),
            exception.finding.strip(),
        )

        if key not in deduplicated:
            deduplicated[key] = exception

    return list(
        deduplicated.values()
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

    normalized_content = normalize_ej_text(
        ej_content
    )

    document_blocks = split_into_document_blocks(
        normalized_content
    )

    grouped_slips = group_blocks_by_slip(
        document_blocks
    )

    add_missing_slip_contexts(
        normalized_content,
        grouped_slips,
    )

    candidate_trails: List[
        Tuple[str, str]
    ] = []

    skipped_slips: List[Dict] = []

    for slip_number, blocks in grouped_slips.items():
        trail, document_types = build_slip_trail(
            slip_number,
            blocks,
        )

        should_review, reason = (
            should_review_with_gemini(
                trail,
                document_types,
            )
        )

        if should_review:
            candidate_trails.append(
                (
                    slip_number,
                    trail,
                )
            )
        else:
            skipped_slips.append(
                {
                    "slip_number": slip_number,
                    "reason": reason,
                }
            )

    batches = create_batches(
        candidate_trails
    )

    all_exceptions: List[
        POSException
    ] = []

    batch_processing_seconds: List[
        float
    ] = []

    if batches:
        client = get_gemini_client()

        for batch_number, batch in enumerate(
            batches,
            start=1,
        ):
            batch_result, elapsed = analyze_batch(
                client=client,
                store=store,
                ej_file_name=ej_file_name,
                batch_number=batch_number,
                batch=batch,
                approved_rules=approved_rules,
            )

            all_exceptions.extend(
                batch_result.exceptions
            )

            batch_processing_seconds.append(
                elapsed
            )

    all_exceptions = deduplicate_exceptions(
        all_exceptions
    )

    analysis_summary = {
        "raw_content_characters": len(
            normalized_content
        ),
        "document_blocks_found": len(
            document_blocks
        ),
        "slips_found": len(
            grouped_slips
        ),
        "candidate_slips": len(
            candidate_trails
        ),
        "production_or_low_signal_slips_skipped": len(
            skipped_slips
        ),
        "gemini_batches_processed": len(
            batches
        ),
        "batch_processing_seconds": (
            batch_processing_seconds
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
# ============================================================

@app.post("/analyze")
def analyze():
    request_started = time.time()

    try:
        payload = request.get_json(
            silent=True
        )

        if not isinstance(payload, dict):
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

        if mode != "analyze_ej":
            return jsonify(
                {
                    "success": False,
                    "error": (
                        f"Unsupported mode: {mode}"
                    ),
                }
            ), 400

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
                    "error": (
                        "ej_content is empty."
                    ),
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

        exceptions, analysis_summary = analyze_ej(
            store=store,
            ej_file_name=ej_file_name,
            ej_content=ej_content,
            approved_rules=approved_rules,
        )

        processing_seconds = round(
            time.time() - request_started,
            3,
        )

        # IMPORTANT:
        # Existing response field names are kept unchanged.
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

                # New additive field only.
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
