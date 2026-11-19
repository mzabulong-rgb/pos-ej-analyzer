import base64
import json
import logging
import os
import re
import time
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Set, Tuple

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
    "REPRINT BILL",
    "INVOICE",
    "FINAL INVOICE",
    "OFFICIAL RECEIPT",
    "SALES INVOICE",
    "BILL CANCELLATION",
    "BILL CANCEL",
    "CANCELLATION",
    "RETURN",
    "REFUND",
    "REVERSAL",
    "CREDIT MEMO",
}

KNOWN_TYPES = PRODUCTION_TYPES | CUSTOMER_TYPES


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
# TEXT HELPERS
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


def decode_ej_content(
    value,
    file_name: str,
) -> str:
    """
    Accept either:

    1. Plain EJ text; or
    2. SharePoint Get file content output such as:
       {
           "$content-type": "application/octet-stream",
           "$content": "base64 text"
       }
    """

    if isinstance(
        value,
        str,
    ):
        return value

    if isinstance(
        value,
        dict,
    ):
        encoded_content = (
            value.get("$content")
            or value.get("content")
        )

        if (
            isinstance(
                encoded_content,
                str,
            )
            and encoded_content
        ):
            try:
                return (
                    base64.b64decode(
                        encoded_content
                    )
                    .decode(
                        "utf-8-sig",
                        errors="replace",
                    )
                )

            except Exception as exc:
                raise ValueError(
                    (
                        "Could not decode ej_content "
                        f"for {file_name}: {exc}"
                    )
                ) from exc

    raise ValueError(
        (
            f"ej_content for {file_name} "
            "must be text or SharePoint file content."
        )
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
        matches = re.findall(
            pattern,
            text,
        )

        for value in matches:
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
    for line in block.splitlines()[:30]:
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
    current_block: List[str] = []

    for line in ej_content.splitlines():
        if (
            is_document_heading(line)
            and current_block
        ):
            completed_block = "\n".join(
                current_block
            ).strip()

            if completed_block:
                blocks.append(
                    completed_block
                )

            current_block = [
                line
            ]

        else:
            current_block.append(
                line
            )

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
    grouped: Dict[
        str,
        List[str],
    ] = {}

    last_slip_number: Optional[str] = None

    for block in blocks:
        slip_numbers = extract_slip_numbers(
            block
        )

        if slip_numbers:
            for slip_number in slip_numbers:
                grouped.setdefault(
                    slip_number,
                    [],
                ).append(
                    block
                )

            last_slip_number = (
                slip_numbers[-1]
            )

            continue

        document_type = classify_document(
            block
        )

        # Customer-facing sections, particularly an INVOICE body,
        # may not repeat the Slip Number after the document heading.
        if (
            document_type in CUSTOMER_TYPES
            and last_slip_number
        ):
            grouped.setdefault(
                last_slip_number,
                [],
            ).append(
                block
            )

    return grouped


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
# TRAIL COMPACTION
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

    kept_lines: List[str] = []

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

        upper_line = (
            stripped_line.upper()
        )

        if any(
            phrase in upper_line
            for phrase in phrases
        ):
            continue

        kept_lines.append(
            stripped_line
        )

    return "\n".join(
        kept_lines
    )


def build_slip_trail(
    slip_number: str,
    blocks: List[str],
) -> Tuple[str, List[str]]:
    compacted_blocks: List[str] = []
    document_types: List[str] = []
    seen_blocks = set()

    for block in blocks:
        cleaned_block = remove_boilerplate(
            block
        ).strip()

        if not cleaned_block:
            continue

        fingerprint = cleaned_block[:800]

        if fingerprint in seen_blocks:
            continue

        seen_blocks.add(
            fingerprint
        )

        document_type = classify_document(
            block
        )

        document_types.append(
            document_type
        )

        compacted_blocks.append(
            (
                f"[DOCUMENT TYPE: {document_type}]\n"
                f"{cleaned_block}"
            )
        )

    trail = (
        f"SLIP NUMBER: {slip_number}\n\n"
        + "\n\n".join(
            compacted_blocks
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
    normalized_types = [
        str(document_type or "").strip().upper()
        for document_type in document_types
        if document_type
    ]

    known_types = [
        document_type
        for document_type in normalized_types
        if document_type != "UNKNOWN"
    ]

    return (
        bool(known_types)
        and all(
            document_type in PRODUCTION_TYPES
            for document_type in known_types
        )
    )


def should_review(
    trail: str,
    document_types: List[str],
) -> Tuple[bool, str]:

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------

    def normalize_screening_text(
        value: str,
    ) -> str:
        return re.sub(
            r"\s+",
            " ",
            str(value or "").strip().upper(),
        )

    def extract_amount(
        value: str,
    ) -> Optional[Decimal]:
        amount_pattern = re.compile(
            r"(?<!\d)"
            r"(?:PHP|P|₱)?\s*"
            r"(-?\d{1,3}(?:,\d{3})*(?:\.\d{2})"
            r"|-?\d+\.\d{2})"
            r"(?!\d)",
            re.IGNORECASE,
        )

        matches = amount_pattern.findall(
            value or ""
        )

        if not matches:
            return None

        try:
            return Decimal(
                matches[-1].replace(",", "")
            )
        except InvalidOperation:
            return None

    def extract_document_amount(
        trail_lines: List[str],
        document_position: int,
        stop_positions: Set[int],
        maximum_lookahead: int = 80,
    ) -> Optional[Decimal]:
        amount_keywords = {
            "TOTAL AMOUNT DUE",
            "GRAND TOTAL",
            "TOTAL AMOUNT",
            "AMOUNT DUE",
            "NET AMOUNT DUE",
            "NET TOTAL",
            "NET AMOUNT",
            "TOTAL DUE",
            "BALANCE DUE",
            "TOTAL",
        }

        excluded_amount_keywords = {
            "GROSS SALES",
            "VATABLE SALES",
            "VAT EXEMPT SALES",
            "VAT AMOUNT",
            "DISCOUNT AMOUNT",
            "SERVICE CHARGE",
            "CHANGE",
            "CASH",
            "GCASH",
            "PAYMAYA",
            "CREDIT CARD",
            "DEBIT CARD",
            "TENDER",
            "PAYMENT",
        }

        end_position = min(
            len(trail_lines),
            document_position
            + maximum_lookahead
            + 1,
        )

        for current_position in range(
            document_position,
            end_position,
        ):
            if (
                current_position
                != document_position
                and current_position
                in stop_positions
            ):
                break

            current_line = trail_lines[
                current_position
            ]

            normalized_line = (
                normalize_screening_text(
                    current_line
                )
            )

            line_amount = extract_amount(
                current_line
            )

            if line_amount is None:
                continue

            if any(
                excluded_keyword
                in normalized_line
                for excluded_keyword
                in excluded_amount_keywords
            ):
                continue

            if any(
                amount_keyword
                in normalized_line
                for amount_keyword
                in amount_keywords
            ):
                return abs(line_amount)

        return None

    def amounts_match(
        first_amount: Decimal,
        second_amount: Decimal,
        tolerance: Decimal = Decimal("0.01"),
    ) -> bool:
        return (
            abs(
                first_amount
                - second_amount
            )
            <= tolerance
        )

    def get_document_block(
        trail_lines: List[str],
        document_position: int,
        marker_positions: List[int],
    ) -> List[str]:
        """
        Gets the contents of a document block.

        The block begins after its [DOCUMENT TYPE: ...]
        marker and ends before the next document marker.
        """

        next_position = next(
            (
                position
                for position
                in marker_positions
                if position
                > document_position
            ),
            len(trail_lines),
        )

        return trail_lines[
            document_position + 1:
            next_position
        ]

    def normalize_bill_content(
        bill_lines: List[str],
    ) -> List[str]:
        """
        Creates a comparable representation of a Print Bill.

        The comparison keeps:
        - items
        - quantities
        - prices
        - discounts
        - charges
        - total amount

        It removes operational details that can change during
        a legitimate unchanged reprint.
        """

        ignored_prefixes = {
            "PRINT BILL",
            "CUSTOMER BILL",
            "REBILL",
            "RE-BILL",
            "REPRINT",
            "BILL REPRINT",
            "REPRINT BILL",
            "SLIP:",
            "SLIP NUMBER:",
            "SLIP NO:",
            "DATE:",
            "TIME:",
            "BUSINESS DATE:",
            "CASHIER:",
            "STAFF:",
            "SERVER:",
            "WAITER:",
            "POS:",
            "POS ID:",
            "TERMINAL:",
            "TRANSACTION:",
            "TRANSACTION NO:",
            "TRANS NO:",
            "TABLE:",
            "TABLE NO:",
            "GUEST:",
            "GUESTS:",
            "COVERS:",
            "PRINTED:",
            "PRINT DATE:",
            "PRINT TIME:",
        }

        normalized_content: List[str] = []

        for line in bill_lines:
            normalized_line = (
                normalize_screening_text(
                    line
                )
            )

            if not normalized_line:
                continue

            if normalized_line.startswith(
                "[SOURCE EJ FILE:"
            ):
                continue

            if normalized_line.startswith(
                "[DOCUMENT TYPE:"
            ):
                continue

            if any(
                normalized_line.startswith(
                    prefix
                )
                for prefix in ignored_prefixes
            ):
                continue

            # Ignore separator lines.
            if set(normalized_line) <= {
                "-",
                "=",
                "_",
                " ",
            }:
                continue

            # Ignore standalone dates.
            if re.fullmatch(
                r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}",
                normalized_line,
            ):
                continue

            # Ignore standalone times.
            if re.fullmatch(
                r"\d{1,2}:\d{2}(?::\d{2})?"
                r"(?:\s*[AP]M)?",
                normalized_line,
            ):
                continue

            normalized_content.append(
                normalized_line
            )

        return normalized_content

    # --------------------------------------------------------
    # NORMALIZE INPUT
    # --------------------------------------------------------

    trail = trail or ""
    lines = trail.splitlines()

    normalized_types = [
        normalize_screening_text(
            document_type
        )
        for document_type
        in document_types
        if document_type
    ]

    # --------------------------------------------------------
    # 1. PRODUCTION-ONLY
    # --------------------------------------------------------

    if is_production_only(
        normalized_types
    ):
        return (
            False,
            "Production-printer records only.",
        )

    # --------------------------------------------------------
    # 2. DOCUMENT TYPES AND REVIEW SIGNALS
    # --------------------------------------------------------

    print_bill_types = {
        "BILL",
        "PRINT BILL",
        "CUSTOMER BILL",
        "REBILL",
        "RE-BILL",
        "REPRINT",
        "BILL REPRINT",
        "REPRINT BILL",
    }

    invoice_types = {
        "INVOICE",
        "FINAL INVOICE",
        "SALES INVOICE",
        "OFFICIAL RECEIPT",
    }

    always_review_document_types = {
        "BILL CANCELLATION",
        "BILL CANCEL",
        "CANCELLATION",
        "RETURN",
        "REFUND",
        "REVERSAL",
        "CREDIT MEMO",
    }

    bill_cancellation_phrases = {
        "THIS CANCELS BILL",
        "THIS CANCELS BILLS",
        "CANCELS BILL NO",
        "CANCELS BILL NUMBER",
    }

    # These trigger review only when they occur
    # after the first customer Print Bill.
    post_bill_irregularity_keywords = {
        "VOID",
        "VOIDED",
        "VOID ITEM",
        "ITEM VOID",
        "DELETE",
        "DELETED",
        "DELETE ITEM",
        "ITEM DELETED",
        "REMOVE",
        "REMOVED",
        "REMOVE ITEM",
        "ITEM REMOVED",
        "CANCEL ITEM",
        "ITEM CANCEL",
        "CANCELLED ITEM",
        "CANCELED ITEM",
        "RETURN",
        "REFUND",
        "REVERSAL",
        "PRICE CHANGE",
        "PRICE OVERRIDE",
        "QUANTITY CHANGE",
        "QTY CHANGE",
        "MANUAL PRICE",
        "OPEN ITEM",
    }

    # --------------------------------------------------------
    # 3. READ DOCUMENT MARKERS
    # --------------------------------------------------------

    document_marker_pattern = re.compile(
        r"^\[DOCUMENT TYPE:\s*(.+?)\]\s*$",
        re.IGNORECASE,
    )

    document_markers: List[
        Tuple[int, str]
    ] = []

    for index, line in enumerate(
        lines
    ):
        marker_match = (
            document_marker_pattern.match(
                line.strip()
            )
        )

        if not marker_match:
            continue

        marker_type = (
            normalize_screening_text(
                marker_match.group(1)
            )
        )

        document_markers.append(
            (
                index,
                marker_type,
            )
        )

    # --------------------------------------------------------
    # 4. IDENTIFY DOCUMENT POSITIONS
    # --------------------------------------------------------

    production_positions = [
        position
        for position, document_type
        in document_markers
        if document_type
        in PRODUCTION_TYPES
    ]

    bill_positions = [
        position
        for position, document_type
        in document_markers
        if document_type
        in print_bill_types
    ]

    invoice_positions = [
        position
        for position, document_type
        in document_markers
        if document_type
        in invoice_types
    ]

    high_priority_positions = [
        position
        for position, document_type
        in document_markers
        if document_type
        in always_review_document_types
    ]

    # --------------------------------------------------------
    # FALLBACK WHEN DOCUMENT MARKERS ARE UNAVAILABLE
    # --------------------------------------------------------

    if not document_markers:
        for index, line in enumerate(
            lines
        ):
            normalized_line = (
                normalize_screening_text(
                    line
                )
            )

            if (
                normalized_line
                in PRODUCTION_TYPES
            ):
                production_positions.append(
                    index
                )

            if (
                normalized_line
                in print_bill_types
            ):
                bill_positions.append(
                    index
                )

            if (
                normalized_line
                in invoice_types
            ):
                invoice_positions.append(
                    index
                )

            if (
                normalized_line
                in always_review_document_types
            ):
                high_priority_positions.append(
                    index
                )

    production_positions = sorted(
        set(production_positions)
    )

    bill_positions = sorted(
        set(bill_positions)
    )

    invoice_positions = sorted(
        set(invoice_positions)
    )

    high_priority_positions = sorted(
        set(high_priority_positions)
    )

    has_production_document = bool(
        production_positions
    )

    has_print_bill = bool(
        bill_positions
    )

    has_invoice = bool(
        invoice_positions
    )

    # --------------------------------------------------------
    # 5. HIGH-PRIORITY DOCUMENTS ALWAYS REQUIRE REVIEW
    # --------------------------------------------------------

    matched_high_priority_type = next(
        (
            document_type
            for document_type
            in normalized_types
            if document_type
            in always_review_document_types
        ),
        None,
    )

    if (
        matched_high_priority_type
        or high_priority_positions
    ):
        return (
            True,
            (
                "High-priority cancellation, "
                "return, refund, reversal, or "
                "credit document was found."
            ),
        )

    # --------------------------------------------------------
    # 6. "THIS CANCELS BILL..." REQUIRES REVIEW
    # --------------------------------------------------------

    for line in lines:
        normalized_line = (
            normalize_screening_text(
                line
            )
        )

        matched_phrase = next(
            (
                phrase
                for phrase
                in bill_cancellation_phrases
                if phrase
                in normalized_line
            ),
            None,
        )

        if matched_phrase:
            return (
                True,
                (
                    "Bill cancellation phrase "
                    f"found: {line.strip()}"
                ),
            )

    # --------------------------------------------------------
    # 7. REQUIRED NORMAL TRANSACTION DOCUMENTS
    # --------------------------------------------------------

    if not has_production_document:
        return (
            True,
            (
                "No production/KOT document "
                "was found."
            ),
        )

    if not has_print_bill:
        return (
            True,
            "No customer Print Bill found.",
        )

    if not has_invoice:
        return (
            True,
            "No final Invoice found.",
        )

    first_bill_position = (
        bill_positions[0]
    )

    final_invoice_position = (
        invoice_positions[-1]
    )

    # --------------------------------------------------------
    # 8. VALIDATE DOCUMENT ORDER
    # --------------------------------------------------------

    if (
        final_invoice_position
        < first_bill_position
    ):
        return (
            True,
            (
                "Final Invoice appeared before "
                "the first Print Bill."
            ),
        )

    bills_before_invoice = [
        position
        for position
        in bill_positions
        if position
        < final_invoice_position
    ]

    if not bills_before_invoice:
        return (
            True,
            (
                "No customer Print Bill was "
                "found before the final Invoice."
            ),
        )

    marker_positions = sorted(
        position
        for position, _
        in document_markers
    )

    # --------------------------------------------------------
    # 9. POST-BILL IRREGULARITY CHECK
    #
    # VOID before the first Print Bill is allowed.
    #
    # VOID, removal, deletion, cancellation, or another
    # configured transaction change after the first Print
    # Bill requires review.
    # --------------------------------------------------------

    next_marker_after_first_bill = next(
        (
            position
            for position
            in marker_positions
            if position
            > first_bill_position
        ),
        final_invoice_position,
    )

    for index in range(
        next_marker_after_first_bill,
        final_invoice_position,
    ):
        normalized_line = (
            normalize_screening_text(
                lines[index]
            )
        )

        matched_keyword = next(
            (
                keyword
                for keyword
                in post_bill_irregularity_keywords
                if keyword
                in normalized_line
            ),
            None,
        )

        if matched_keyword:
            return (
                True,
                (
                    "Transaction irregularity "
                    "occurred after the first "
                    "Print Bill: "
                    f"{matched_keyword}. "
                    "Trail entry: "
                    f"{lines[index].strip()}"
                ),
            )

    # --------------------------------------------------------
    # 10. COMPARE MULTIPLE PRINT BILLS
    #
    # Identical Print Bill reprints are not automatically
    # flagged.
    #
    # A difference in items, quantities, prices, discounts,
    # charges, or totals requires review.
    # --------------------------------------------------------

    if len(
        bills_before_invoice
    ) >= 2:
        first_bill_lines = (
            get_document_block(
                trail_lines=lines,
                document_position=(
                    bills_before_invoice[0]
                ),
                marker_positions=(
                    marker_positions
                ),
            )
        )

        first_bill_content = (
            normalize_bill_content(
                first_bill_lines
            )
        )

        if not first_bill_content:
            return (
                True,
                (
                    "Multiple Print Bills were "
                    "found, but the first Bill "
                    "contents could not be "
                    "reliably compared."
                ),
            )

        for current_bill_position in (
            bills_before_invoice[1:]
        ):
            current_bill_lines = (
                get_document_block(
                    trail_lines=lines,
                    document_position=(
                        current_bill_position
                    ),
                    marker_positions=(
                        marker_positions
                    ),
                )
            )

            current_bill_content = (
                normalize_bill_content(
                    current_bill_lines
                )
            )

            if not current_bill_content:
                return (
                    True,
                    (
                        "Multiple Print Bills were "
                        "found, but a later Bill's "
                        "contents could not be "
                        "reliably compared."
                    ),
                )

            if (
                current_bill_content
                != first_bill_content
            ):
                return (
                    True,
                    (
                        "Multiple customer Print "
                        "Bills were found and the "
                        "Bill contents changed "
                        "between prints."
                    ),
                )

    # --------------------------------------------------------
    # 11. EXTRACT FINAL BILL AND INVOICE AMOUNTS
    # --------------------------------------------------------

    final_bill_position = (
        bills_before_invoice[-1]
    )

    all_document_positions = set(
        production_positions
        + bill_positions
        + invoice_positions
        + high_priority_positions
    )

    bill_amount = (
        extract_document_amount(
            trail_lines=lines,
            document_position=(
                final_bill_position
            ),
            stop_positions=(
                all_document_positions
            ),
        )
    )

    invoice_amount = (
        extract_document_amount(
            trail_lines=lines,
            document_position=(
                final_invoice_position
            ),
            stop_positions=(
                all_document_positions
            ),
        )
    )

    if bill_amount is None:
        return (
            True,
            (
                "Final Print Bill found, but "
                "its total amount could not be "
                "reliably identified."
            ),
        )

    if invoice_amount is None:
        return (
            True,
            (
                "Invoice found, but its final "
                "Total Amount Due could not be "
                "reliably identified."
            ),
        )

    # --------------------------------------------------------
    # 12. FINAL BILL MUST MATCH FINAL INVOICE
    # --------------------------------------------------------

    if not amounts_match(
        bill_amount,
        invoice_amount,
    ):
        return (
            True,
            (
                "Final Print Bill and Invoice "
                "amounts do not match: "
                f"Bill={bill_amount:.2f}, "
                f"Invoice={invoice_amount:.2f}."
            ),
        )

    # --------------------------------------------------------
    # 13. NORMAL TRANSACTION
    # --------------------------------------------------------

    if len(
        bills_before_invoice
    ) >= 2:
        return (
            False,
            (
                "Unchanged Print Bill reprint: "
                "all customer Print Bills have "
                "matching contents, the final "
                "Bill matches the Invoice at "
                f"{bill_amount:.2f}, and no "
                "post-bill irregularity was "
                "found."
            ),
        )

    return (
        False,
        (
            "Normal transaction: production/KOT, "
            "one Print Bill, and final Invoice "
            "were found; Print Bill and Invoice "
            "amounts match at "
            f"{bill_amount:.2f}; no transaction "
            "irregularity occurred after the "
            "Print Bill."
        ),
    )

# ============================================================
# GEMINI BATCHING
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

    current_batch: List[
        Tuple[str, str]
    ] = []

    current_character_count = 0

    for (
        slip_number,
        trail,
    ) in candidate_trails:
        trail_character_count = len(
            trail
        )

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
# GEMINI SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a POS Electronic Journal transaction exception analyst.

You will receive complete transaction trails grouped by Slip Number.

Evaluate every supplied Slip independently.

A Slip trail may contain records from more than one SOURCE EJ FILE.

When the same Slip Number appears across multiple EJ files:

1. Treat all records with that exact Slip Number as one complete
   transaction trail.
2. Reconstruct the activity chronologically using all available
   timestamps.
3. Do not treat a source-file boundary as the end of the transaction.
4. Consider all bills, item changes, cancellations, discounts,
   tenders, invoices, returns, and refunds across all source EJ files.
5. Mention relevant source EJ file names in evidence_trail when useful.

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

1. Multiple customer Print Bills where the amount, items, quantities,
   prices, discounts, charges, or other material transaction details
   changed between prints.
   Do not flag multiple customer Print Bills solely because the Bill
   was reprinted when the punched items, quantities, prices, discounts,
   charges, and total amount remained unchanged.
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
14. A customer Print Bill remaining outstanding for an unusually long
    period when the supplied evidence supports that conclusion.
15. Other materially inconsistent transaction sequences that warrant
    human investigation.

============================================================
DO NOT FLAG SOLELY BECAUSE
============================================================

Do not flag solely because:

- the same Slip Number appears repeatedly;
- the same Slip Number appears in more than one EJ file;
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
The amount shown on the FIRST identifiable CUSTOMER Print Bill.

Use only an actual customer-facing BILL, PRINT BILL, CUSTOMER BILL,
REBILL, or BILL REPRINT.

Never use a production-printer document as Initial Bill Amount.


Rebilled Amount:
The amount shown on the LAST identifiable CUSTOMER Print Bill that
occurs after Initial Bill Amount and before final settlement.

Only populate Rebilled Amount when a later customer Print Bill
actually exists.

If there is only one customer Print Bill, return null for
Rebilled Amount.

Do not copy Initial Bill Amount into Rebilled Amount merely because
there was no later customer bill.


Final Invoice Amount:
When an INVOICE is available, identify the FINAL CUSTOMER PAYABLE
AMOUNT shown on the invoice after discounts, VAT treatment, service
charge, and other applicable adjustments.

Prefer amounts labeled:

- Total Amount Due
- Amount Due
- Grand Total
- Net Amount Due
- Total Due

If Total Amount Due exists, use it as Final Invoice Amount.

Do not use Gross Sales when a later Total Amount Due exists.

Do not use these individually as Final Invoice Amount:

- VATable Sales
- VAT Exempt Sales
- VAT Amount
- Discount Amount
- Service Charge
- Tender or payment amount

The Final Invoice Amount must be returned as a positive numeric value.

Example:

Gross Sales                    1,500.00
Less: Discount                  -100.00
Service Charge                   140.00
Total Amount Due               1,540.00
PAYMAYA                       -1,540.00

Return:

final_invoice_amount = 1540.00


Amount Difference:
If two or more customer Print Bills exist:

amount_difference =
Rebilled Amount minus Initial Bill Amount.

If only one customer Print Bill exists and Final Invoice Amount exists:

amount_difference =
Final Invoice Amount minus Initial Bill Amount.

A positive value means the customer-facing amount increased.

A negative value means the customer-facing amount decreased.

Zero means there was no applicable amount change.

If the required amounts cannot be reliably identified, return null.


Bill Amount Trail:
Preserve every identifiable CUSTOMER Print Bill amount in chronological
order.

Do not include production-printer amounts.

Do not include Final Invoice Amount unless the invoice is explicitly
also identified as a customer Print Bill.


Number of Bill Prints:
Count only identifiable CUSTOMER Print Bills.

Do not count production-printer documents.

Do not count the final INVOICE unless it is explicitly also identified
as a customer Print Bill.

============================================================
FLAG CATEGORIES
============================================================

Use one or more of these standardized Flag Category values when the
identified exception reasonably fits an existing category:

- Multiple Print Bill
- Amount Change
- Item Removal
- Item Addition
- Bill Cancellation
- Repeated Bill Cancellation
- Rebill
- Return/Refund
- Multiple Discount
- Discount Change
- VAT Exempt + Discount
- Tender Change
- Reprint
- Post - Settlement Change
- Unusual Transaction Sequence
- Long Outstanding Print Bill

These categories are classification labels only.

Do NOT limit exception detection to this category list.

For a known pattern, preserve the exact SharePoint spelling,
capitalization, spacing, and punctuation.

Do not return internal codes such as:

ITEM_ADDED_AFTER_BILL

Return the SharePoint-compatible value instead:

Item Addition

If a material irregularity does not reasonably fit an existing
category:

1. Do not ignore it.
2. Use flag_categories = ["Unusual Transaction Sequence"].
3. Describe the specific new pattern in rule_triggered.
4. Explain the issue in finding.
5. Preserve chronological evidence in evidence_trail.

Multiple categories may be returned when multiple distinct patterns
are supported by the same Slip trail.

============================================================
EVIDENCE
============================================================

Every exception must be supported by specific evidence from the
supplied Slip trail.

Explain the relevant sequence chronologically.

Do not invent Slip Numbers, transaction numbers, amounts, timestamps,
items, discounts, documents, tenders, invoices, cancellations,
returns, refunds, or source file names.

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
    source_description: str,
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

    combined_trails = "\n\n".join(
        trail
        for (
            _,
            trail,
        ) in batch
    )

    user_prompt = f"""
STORE:
{store}

SOURCE EJ FILES:
{source_description}

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
{combined_trails}
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
                "Gemini returned an empty response "
                f"for batch {batch_number}."
            )
        )

    try:
        result = (
            EJAnalysis.model_validate_json(
                response.text
            )
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
# DEDUPLICATION
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
# SHARED GROUPED-SLIP ANALYSIS
# ============================================================

def analyze_grouped_slips(
    store: str,
    source_description: str,
    grouped_slips: Dict[
        str,
        List[str],
    ],
    approved_rules=None,
) -> Tuple[
    List[POSException],
    Dict,
]:
    if approved_rules is None:
        approved_rules = []

    candidate_trails: List[
        Tuple[str, str]
    ] = []

    skipped_slips = 0

    for (
        slip_number,
        slip_blocks,
    ) in grouped_slips.items():
        (
            trail,
            document_types,
        ) = build_slip_trail(
            slip_number,
            slip_blocks,
        )

        (
            should_analyze,
            _,
        ) = should_review(
            trail,
            document_types,
        )

        if should_analyze:
            candidate_trails.append(
                (
                    slip_number,
                    trail,
                )
            )

        else:
            skipped_slips += 1

    batches = create_batches(
        candidate_trails
    )

    logger.info(
        "EJ batch screening complete: slips_found=%s candidate_slips=%s "
        "skipped_slips=%s gemini_batches=%s",
        len(grouped_slips),
        len(candidate_trails),
        skipped_slips,
        len(batches),
    )

    all_exceptions: List[
        POSException
    ] = []

    batch_processing_seconds: List[
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
            logger.info(
                "Starting Gemini batch %s of %s with %s slips",
                batch_number,
                len(batches),
                len(batch),
            )

            (
                batch_result,
                elapsed,
            ) = analyze_batch(
                client=client,
                store=store,
                source_description=(
                    source_description
                ),
                batch_number=batch_number,
                batch=batch,
                approved_rules=approved_rules,
            )

            logger.info(
                "Finished Gemini batch %s of %s in %.3f seconds",
                batch_number,
                len(batches),
                elapsed,
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
        "slips_found": len(
            grouped_slips
        ),
        "candidate_slips": len(
            candidate_trails
        ),
        "production_or_low_signal_slips_skipped": (
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
# SINGLE-FILE ANALYSIS
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

    normalized_content = normalize_text(
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

    (
        exceptions,
        analysis_summary,
    ) = analyze_grouped_slips(
        store=store,
        source_description=ej_file_name,
        grouped_slips=grouped_slips,
        approved_rules=approved_rules,
    )

    analysis_summary.update(
        {
            "files_received": 1,
            "ej_file_names": [
                ej_file_name
            ],
            "raw_content_characters": len(
                normalized_content
            ),
            "document_blocks_found": len(
                document_blocks
            ),
            "cross_file_slips": 0,
        }
    )

    return (
        exceptions,
        analysis_summary,
    )


# ============================================================
# MULTI-FILE / CROSS-FILE ANALYSIS
# ============================================================

def analyze_ej_batch(
    store: str,
    ej_files: List[Dict],
    approved_rules=None,
) -> Tuple[
    List[POSException],
    Dict,
    str,
]:
    if approved_rules is None:
        approved_rules = []

    combined_grouped_slips: Dict[
        str,
        List[str],
    ] = {}

    slip_source_files: Dict[
        str,
        Set[str],
    ] = {}

    ej_file_names: List[str] = []

    total_raw_characters = 0
    total_document_blocks = 0

    for (
        file_index,
        ej_file,
    ) in enumerate(
        ej_files,
        start=1,
    ):
        if not isinstance(
            ej_file,
            dict,
        ):
            raise ValueError(
                (
                    f"ej_files item {file_index} "
                    "must be an object."
                )
            )

        ej_file_name = str(
            ej_file.get(
                "ej_file_name",
                "",
            )
        ).strip()

        if not ej_file_name:
            raise ValueError(
                (
                    f"ej_files item {file_index} "
                    "is missing ej_file_name."
                )
            )

        ej_content = decode_ej_content(
            ej_file.get(
                "ej_content",
                "",
            ),
            ej_file_name,
        )

        if not ej_content.strip():
            raise ValueError(
                (
                    f"ej_content for {ej_file_name} "
                    "is empty."
                )
            )

        ej_file_names.append(
            ej_file_name
        )

        normalized_content = normalize_text(
            ej_content
        )

        total_raw_characters += len(
            normalized_content
        )

        document_blocks = split_into_document_blocks(
            normalized_content
        )

        total_document_blocks += len(
            document_blocks
        )

        grouped_slips_in_file = (
            group_blocks_by_slip(
                document_blocks
            )
        )

        add_missing_slip_contexts(
            normalized_content,
            grouped_slips_in_file,
        )

        for (
            slip_number,
            slip_blocks,
        ) in grouped_slips_in_file.items():
            combined_grouped_slips.setdefault(
                slip_number,
                [],
            )

            slip_source_files.setdefault(
                slip_number,
                set(),
            ).add(
                ej_file_name
            )

            for block in slip_blocks:
                source_marked_block = (
                    f"[SOURCE EJ FILE: {ej_file_name}]\n"
                    f"{block}"
                )

                combined_grouped_slips[
                    slip_number
                ].append(
                    source_marked_block
                )

    source_description = "; ".join(
        ej_file_names
    )

    (
        exceptions,
        analysis_summary,
    ) = analyze_grouped_slips(
        store=store,
        source_description=source_description,
        grouped_slips=combined_grouped_slips,
        approved_rules=approved_rules,
    )

    cross_file_slips = sum(
        1
        for source_files in (
            slip_source_files.values()
        )
        if len(source_files) > 1
    )

    analysis_summary.update(
        {
            "files_received": len(
                ej_file_names
            ),
            "ej_file_names": (
                ej_file_names
            ),
            "raw_content_characters": (
                total_raw_characters
            ),
            "document_blocks_found": (
                total_document_blocks
            ),
            "cross_file_slips": (
                cross_file_slips
            ),
        }
    )

    return (
        exceptions,
        analysis_summary,
        source_description,
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
            "supported_modes": [
                "analyze_ej",
                "analyze_ej_batch",
            ],
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
            "supported_modes": [
                "analyze_ej",
                "analyze_ej_batch",
            ],
        }
    )


# ============================================================
# ANALYZE ENDPOINT
# ============================================================

@app.post("/")
@app.post("/analyze")
def analyze():
    request_started = time.time()

    mode = "analyze_ej"

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

        approved_rules = payload.get(
            "approved_rules",
            [],
        )

        if mode not in {
            "analyze_ej",
            "analyze_ej_batch",
        }:
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

        if mode == "analyze_ej":
            ej_file_name = str(
                payload.get(
                    "ej_file_name",
                    "",
                )
            ).strip()

            if not ej_file_name:
                return jsonify(
                    {
                        "success": False,
                        "error": (
                            "ej_file_name is required."
                        ),
                    }
                ), 400

            try:
                ej_content = decode_ej_content(
                    payload.get(
                        "ej_content",
                        "",
                    ),
                    ej_file_name,
                )

            except ValueError as exc:
                return jsonify(
                    {
                        "success": False,
                        "error": str(exc),
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

            (
                exceptions,
                analysis_summary,
            ) = analyze_ej(
                store=store,
                ej_file_name=ej_file_name,
                ej_content=ej_content,
                approved_rules=approved_rules,
            )

            response_file_name = (
                ej_file_name
            )

        else:
            ej_files = payload.get(
                "ej_files",
                [],
            )

            if not isinstance(
                ej_files,
                list,
            ):
                return jsonify(
                    {
                        "success": False,
                        "error": (
                            "ej_files must be an array."
                        ),
                    }
                ), 400

            if not ej_files:
                return jsonify(
                    {
                        "success": False,
                        "error": (
                            "ej_files is empty."
                        ),
                    }
                ), 400

            (
                exceptions,
                analysis_summary,
                response_file_name,
            ) = analyze_ej_batch(
                store=store,
                ej_files=ej_files,
                approved_rules=approved_rules,
            )

        processing_seconds = round(
            time.time() - request_started,
            3,
        )

        return jsonify(
            {
                "success": True,
                "mode": mode,
                "model": GEMINI_MODEL,
                "store": store,
                "ej_file_name": (
                    response_file_name
                ),
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
                "mode": mode,
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
