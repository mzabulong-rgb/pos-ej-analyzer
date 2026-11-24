import base64
import gzip
import json
import logging
import os
import re
import time
import uuid
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Set, Tuple

from flask import Flask, jsonify, request
from google import genai
from google.genai import types
from google.cloud import firestore
from google.cloud import storage
from google.cloud import tasks_v2
from google.protobuf import duration_pb2
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

GEMINI_BATCH_MAX_SLIPS = int(
    os.environ.get(
        "GEMINI_BATCH_MAX_SLIPS",
        "8",
    )
)

GOOGLE_CLOUD_PROJECT = os.environ.get(
    "GOOGLE_CLOUD_PROJECT",
    "tbg-risk-report-512405",
).strip()

TASK_LOCATION = os.environ.get(
    "TASK_LOCATION",
    "asia-southeast1",
).strip()

TASK_QUEUE = os.environ.get(
    "TASK_QUEUE",
    "pos-ej-analysis",
).strip()

JOB_BUCKET = os.environ.get(
    "JOB_BUCKET",
    "tbg-risk-report-512405-pos-ej-jobs",
).strip()

SERVICE_URL = os.environ.get(
    "SERVICE_URL",
    "https://pos-ej-analyzer-502375766659.asia-southeast1.run.app",
).strip().rstrip("/")

TASK_SERVICE_ACCOUNT = os.environ.get(
    "TASK_SERVICE_ACCOUNT",
    "",
).strip()

WORKER_TOKEN = os.environ.get(
    "WORKER_TOKEN",
    "",
).strip()

TASK_DISPATCH_DEADLINE_SECONDS = int(
    os.environ.get(
        "TASK_DISPATCH_DEADLINE_SECONDS",
        "1800",
    )
)

JOB_COLLECTION = os.environ.get(
    "JOB_COLLECTION",
    "pos_ej_analysis_jobs",
).strip()

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
    "PASTRY",
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

SLIP_LINE_PATTERN = re.compile(
    r"^\s*(?:SLIP|SLIP\s+NUMBER|SLIP\s+NO\.?)\s*:\s*"
    r"([A-Z0-9][A-Z0-9_-]{3,49})\s*$",
    re.IGNORECASE,
)


def normalize_slip_number(
    value: str,
) -> str:
    """
    Normalize the Slip Number without changing its identity.
    """

    return re.sub(
        r"[^A-Z0-9_-]",
        "",
        str(value or "").strip().upper(),
    )


def extract_slip_numbers(
    text: str,
) -> List[str]:
    """
    Extract transaction Slip Numbers only from explicit:

    Slip:
    Slip Number:
    Slip No.:

    Bill Slip Ref. No. is not a transaction Slip Number.
    """

    found: List[str] = []

    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()

        # Prevent Bill Slip Ref. No. from being treated
        # as the transaction key.
        if re.match(
            r"^\s*BILL\s+SLIP",
            line,
            re.IGNORECASE,
        ):
            continue

        match = SLIP_LINE_PATTERN.match(
            line
        )

        if not match:
            continue

        slip_number = normalize_slip_number(
            match.group(1)
        )

        if (
            slip_number
            and slip_number not in found
        ):
            found.append(
                slip_number
            )

    return found


def extract_unique_slip_number(
    text: str,
) -> Optional[str]:
    """
    Return a Slip Number only when the complete POS document
    contains exactly one unique explicit Slip Number.

    A document containing more than one Slip Number is unsafe
    and will not be automatically grouped.
    """

    slip_numbers = extract_slip_numbers(
        text
    )

    if len(slip_numbers) == 1:
        return slip_numbers[0]

    if len(slip_numbers) > 1:
        logger.warning(
            (
                "POS document was not assigned because it "
                "contains multiple Slip Numbers: %s"
            ),
            slip_numbers,
        )

    return None
    
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
    """
    Classify one complete POS document.

    A standalone uppercase INVOICE heading identifies an
    Invoice document.

    This only classifies the document type. It does not
    establish transaction ownership.

    The explicit Slip: value remains the only transaction key.
    """

    block_lines = block.splitlines()

    # The INVOICE heading may appear below its metadata,
    # including Inv. No., Slip:, staff, and transaction number.
    has_invoice_heading = any(
        line.strip() == "INVOICE"
        for line in block_lines
    )

    if has_invoice_heading:
        return "INVOICE"

    for line in block_lines[:30]:
        title = normalize_title(
            line
        )

        if title in KNOWN_TYPES:
            return title

    return "UNKNOWN"


def split_into_document_blocks(
    ej_content: str,
) -> List[str]:
    """
    Split EJ content into complete POS document blocks.

    Production nd customer Bills continue to use
    their existing uppercase document headings as boundaries.

    An Invoice is handled specially because the actual EJ
    format places its metadata before the uppercase INVOICE
    heading:

    Transaction Date:
    Printed Date:
    Inv. No.:
    Slip:
    Bill Slip Ref. No.:
    Staff:
    Trans.:
    ...
    INVOICE
    ...
    Total Amount Due
    Tender

    The entire Invoice must stay in one block so its explicit
    Slip: remains attached to the INVOICE body, final amount,
    and tender.

    Transaction Date is used only to detect the beginning of
    the Invoice document. It is never used as a transaction
    key.

    Slip: remains the only transaction-trail key.
    """

    blocks: List[str] = []
    current_block: List[str] = []

    current_block_is_invoice = False

    for line in ej_content.splitlines():
        normalized_line = normalize_title(
            line
        )

        # ----------------------------------------------------
        # START OF AN INVOICE DOCUMENT
        #
        # This detects only the document boundary.
        #
        # It does not associate the Invoice with a transaction.
        # Transaction ownership will still be determined only
        # by the explicit Slip: value.
        # ----------------------------------------------------

        starts_invoice_document = bool(
            re.match(
                r"^\s*TRANSACTION\s+DATE\s*:",
                line,
                re.IGNORECASE,
            )
        )

        if starts_invoice_document:
            if current_block:
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

            current_block_is_invoice = True

            continue

        # ----------------------------------------------------
        # KEEP THE UPPERCASE INVOICE HEADING IN THE SAME BLOCK
        #
        # Do not create another document at INVOICE because
        # the explicit Slip: appears above this heading.
        # ----------------------------------------------------

        if (
            normalized_line == "INVOICE"
            and current_block_is_invoice
        ):
            current_block.append(
                line
            )

            continue

        # ----------------------------------------------------
        # NORMAL DOCUMENT HEADINGS
        #
        # BAR, WINDOW, NACHO, KITCHEN, PANTRY, PASTRY,
        # BILL, BILL CANCELLATION, RETURN, REFUND, and
        # other existing document types continue to split
        # normally using the existing KNOWN_TYPES rules.
        # ----------------------------------------------------

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

            current_block_is_invoice = False

        else:
            current_block.append(
                line
            )

    # --------------------------------------------------------
    # SAVE THE FINAL DOCUMENT BLOCK
    # --------------------------------------------------------

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
    """
    Group complete POS documents using Slip Number as the only
    transaction-trail key.

    Examples:

    KOT Slip 1111
    WINDOW Slip 1111
    BILL Slip 1111
    INVOICE Slip 1111
    = one transaction trail for Slip 1111

    KOT Slip 1111
    INVOICE Slip 1112
    = two separate transaction trails

    A document is never grouped based on:
    - the preceding Slip
    - Bill Slip Ref. No.
    - invoice number
    - transaction number
    - amount
    - table
    - staff
    - items
    - timestamp
    - physical proximity in the EJ
    """

    grouped: Dict[
        str,
        List[str],
    ] = {}

    for (
        block_index,
        block,
    ) in enumerate(
        blocks,
        start=1,
    ):
        slip_number = (
            extract_unique_slip_number(
                block
            )
        )

        if not slip_number:
            logger.warning(
                (
                    "POS document block %s was not assigned "
                    "because it does not contain exactly one "
                    "explicit Slip Number."
                ),
                block_index,
            )

            continue

        grouped.setdefault(
            slip_number,
            [],
        ).append(
            block
        )

    return grouped

def add_missing_slip_contexts(
    ej_content: str,
    grouped: Dict[str, List[str]],
) -> None:
    """
    Proximity-based context recovery is intentionally disabled.

    Transaction grouping must use Slip Number only.

    A document belonging to another Slip Number must never be
    attached based on physical proximity in the EJ file.

    This function is retained only so existing callers do not
    need to be changed.
    """

    del ej_content
    del grouped

    return None
    
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
    """
    Build one complete transaction trail for one Slip Number.

    KOT, WINDOW, PANTRY, BAR, NACHO, BILL, REBILL,
    CANCELLATION, and INVOICE documents may all be included,
    but only when each complete document explicitly contains
    the same Slip Number.
    """

    target_slip = normalize_slip_number(
        slip_number
    )

    compacted_blocks: List[str] = []
    document_types: List[str] = []
    seen_blocks = set()

    for block in blocks:
        block_slip = (
            extract_unique_slip_number(
                block
            )
        )

        # Final protection against cross-slip contamination.
        if block_slip != target_slip:
            logger.error(
                (
                    "Cross-slip document rejected. "
                    "Target Slip=%s; document Slip=%s."
                ),
                target_slip,
                block_slip,
            )

            continue

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
                f"[TRANSACTION SLIP: {target_slip}]\n"
                f"[DOCUMENT TYPE: {document_type}]\n"
                f"{cleaned_block}"
            )
        )

    trail = (
        "============================================================\n"
        f"PRIMARY SLIP NUMBER: {target_slip}\n"
        "TRANSACTION KEY: SLIP NUMBER ONLY\n"
        "Every document below explicitly contains this exact "
        "Slip Number.\n"
        "============================================================\n\n"
        + "\n\n".join(
            compacted_blocks
        )
        + "\n\n"
        "============================================================\n"
        f"END PRIMARY SLIP NUMBER: {target_slip}\n"
        "============================================================"
    )

    return (
        trail,
        document_types,
    )


# ============================================================
# SAME-SLIP INVOICE DATA EXTRACTION
# ============================================================

def extract_same_slip_invoice_data(
    primary_slip_number: str,
    blocks: List[str],
) -> Dict:
    """
    Extract Invoice information only from the document whose
    explicit Slip: exactly matches the Primary Slip.

    Mandatory order:

    1. Read the document's explicit Slip:
    2. Compare it with the Primary Slip.
    3. Reject the document if Slip does not match.
    4. Only after Slip matches, check for uppercase INVOICE.
    5. Extract Inv. No., Total Amount Due, and final tender
       from that same document.

    Slip: is the only transaction key.

    Inv. No., Bill Slip Ref. No., transaction number, date,
    time, table, staff, amount, tender, and document position
    are never used to associate the Invoice with a transaction.
    """

    primary_slip = normalize_slip_number(
        primary_slip_number
    )

    result = {
        "invoice_number": "",
        "final_invoice_amount": None,
        "final_tender": "",
    }

    matching_invoice_blocks: List[str] = []

    for block in blocks:

        # ----------------------------------------------------
        # 1. READ THE EXPLICIT Slip: FIRST
        # ----------------------------------------------------

        document_slip = (
            extract_unique_slip_number(
                block
            )
        )

        # ----------------------------------------------------
        # 2. ENFORCE THE ONLY TRANSACTION KEY
        #
        # Reject the entire document before inspecting
        # INVOICE, Inv. No., amount, or tender.
        # ----------------------------------------------------

        if document_slip != primary_slip:
            continue

        # ----------------------------------------------------
        # 3. ONLY AFTER Slip MATCHES, CHECK FOR INVOICE
        # ----------------------------------------------------

        contains_invoice_heading = any(
            line.strip() == "INVOICE"
            for line in block.splitlines()
        )

        if not contains_invoice_heading:
            continue

        matching_invoice_blocks.append(
            block
        )

    if not matching_invoice_blocks:
        logger.info(
            (
                "No same-Slip INVOICE found. "
                "Primary Slip=%s."
            ),
            primary_slip,
        )

        return result

    # Use the last INVOICE carrying this exact Slip:.
    invoice_block = (
        matching_invoice_blocks[-1]
    )

    # --------------------------------------------------------
    # INVOICE NUMBER
    #
    # Inv. No. is extracted only after the Slip match.
    # It is not a transaction key.
    #
    # Example:
    # Inv. No.: 0-0000000000000029381
    # --------------------------------------------------------

    invoice_number_match = re.search(
        r"(?im)^\s*INV\.?\s*NO\.?\s*:\s*"
        r"([A-Z0-9_-]+)\s*$",
        invoice_block,
    )

    if invoice_number_match:
        result["invoice_number"] = (
            invoice_number_match
            .group(1)
            .strip()
        )

    # --------------------------------------------------------
    # FINAL INVOICE AMOUNT
    #
    # Example:
    # Total Amount Due          1,840.89
    # --------------------------------------------------------

    total_amount_due_match = re.search(
        r"(?im)^\s*TOTAL\s+AMOUNT\s+DUE\s*:?\s*"
        r"(?:PHP|P|₱)?\s*"
        r"(-?\d[\d,]*\.\d{2})\s*$",
        invoice_block,
    )

    if total_amount_due_match:
        try:
            result["final_invoice_amount"] = abs(
                float(
                    total_amount_due_match
                    .group(1)
                    .replace(",", "")
                )
            )

        except ValueError:
            result["final_invoice_amount"] = None

    # --------------------------------------------------------
    # FINAL TENDER
    #
    # Tender is extracted only after Total Amount Due
    # inside the exact same-Slip INVOICE document.
    # --------------------------------------------------------

    tender_search_text = ""

    if total_amount_due_match:
        tender_search_text = invoice_block[
            total_amount_due_match.end():
        ]

    tender_patterns = [
        (
            "Cash",
            r"(?im)^\s*CASH\s*:?\s*"
            r"-?\s*(?:PHP|P|₱)?\s*"
            r"\d[\d,]*\.\d{2}\s*$",
        ),
        (
            "PayMaya",
            r"(?im)^\s*PAYMAYA\s*:?\s*"
            r"-?\s*(?:PHP|P|₱)?\s*"
            r"\d[\d,]*\.\d{2}\s*$",
        ),
        (
            "GCash",
            r"(?im)^\s*GCASH\s*:?\s*"
            r"-?\s*(?:PHP|P|₱)?\s*"
            r"\d[\d,]*\.\d{2}\s*$",
        ),
        (
            "Credit Card",
            r"(?im)^\s*(?:CREDIT\s*CARD|VISA|MASTERCARD)"
            r"\s*:?\s*-?\s*(?:PHP|P|₱)?\s*"
            r"\d[\d,]*\.\d{2}\s*$",
        ),
        (
            "Debit Card",
            r"(?im)^\s*DEBIT\s*CARD\s*:?\s*"
            r"-?\s*(?:PHP|P|₱)?\s*"
            r"\d[\d,]*\.\d{2}\s*$",
        ),
    ]

    for (
        tender_name,
        tender_pattern,
    ) in tender_patterns:
        if re.search(
            tender_pattern,
            tender_search_text,
        ):
            result["final_tender"] = (
                tender_name
            )

            break

    logger.info(
        (
            "Same-Slip INVOICE extracted: "
            "PrimarySlip=%s InvoiceNumber=%s "
            "FinalInvoiceAmount=%s FinalTender=%s"
        ),
        primary_slip,
        result["invoice_number"],
        result["final_invoice_amount"],
        result["final_tender"],
    )

    return result


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
    """
    Create multi-Slip Gemini batches while preserving strict
    Slip Number transaction boundaries.

    Each batch is limited by:

    1. GEMINI_BATCH_MAX_CHARS; and
    2. GEMINI_BATCH_MAX_SLIPS.

    Each Slip trail remains isolated and explicitly identifies
    its own PRIMARY SLIP NUMBER.
    """

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
        normalized_slip = (
            normalize_slip_number(
                slip_number
            )
        )

        trail_character_count = len(
            trail
        )

        exceeds_character_limit = (
            current_batch
            and (
                current_character_count
                + trail_character_count
                > GEMINI_BATCH_MAX_CHARS
            )
        )

        exceeds_slip_limit = (
            current_batch
            and len(
                current_batch
            ) >= GEMINI_BATCH_MAX_SLIPS
        )

        if (
            exceeds_character_limit
            or exceeds_slip_limit
        ):
            batches.append(
                current_batch
            )

            current_batch = []
            current_character_count = 0

        current_batch.append(
            (
                normalized_slip,
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

============================================================
ABSOLUTE TRANSACTION-ID RULE
============================================================

THE VALUE PRINTED AFTER "Slip:" IS THE ONLY TRANSACTION KEY.

Example:

Slip: 000001018B000026037

Transaction Key:
000001018B000026037

This rule applies to EVERY document in EVERY supplied EJ file.

Do not determine transaction ownership using any other field.

NEVER use any of the following as a transaction key:

- Bill Slip No.
- Bill Slip Ref. No.
- Invoice Number
- Transaction Number
- Date
- Business Date
- Printed Date
- Time
- Transaction Time
- Printed Time
- Settlement Time
- POS ID
- Staff
- Cashier
- Table Number
- Guest Number
- Amount
- Tender
- Items
- Discounts
- EJ file name
- document position
- adjacent documents
- similar values
- consecutive transactions

DATE AND TIME ARE NEVER TRANSACTION KEYS.

Two documents occurring seconds or minutes apart do not belong to the
same transaction unless their explicit "Slip:" values are identical.

Two adjacent documents do not belong to the same transaction unless
their explicit "Slip:" values are identical.

Similar amounts, items, staff, tables, POS IDs, tenders, dates, or
times do not establish transaction identity.

ONLY THE EXACT VALUE AFTER "Slip:" ESTABLISHES TRANSACTION IDENTITY.

============================================================
DOCUMENT OWNERSHIP
============================================================

For EVERY document in EVERY supplied EJ file:

1. Locate that document's explicit line beginning with:

   Slip:

2. Extract the value written after "Slip:".

3. That Slip value determines which transaction owns the entire
   document.

4. A document may be used for the transaction currently being analyzed
   ONLY when its explicit Slip value exactly matches the PRIMARY SLIP
   NUMBER.

Example:

PRIMARY SLIP NUMBER:
000001018B000026033

Document A:
Slip: 000001018B000026033
INCLUDE.

Document B:
Slip: 000001018B000026033
INCLUDE.

Document C:
Slip: 000001018B000026037
DO NOT INCLUDE.

Document C belongs to a completely different transaction.

No information from Document C may be used when analyzing
Slip 000001018B000026033.

============================================================
CRITICAL INVOICE RULE
============================================================

An INVOICE belongs to a transaction ONLY when the Invoice document's
explicit "Slip:" value exactly matches the PRIMARY SLIP NUMBER.

Never select or associate an Invoice because the Invoice:

- appears immediately after a Bill;
- is the nearest Invoice;
- has the closest timestamp;
- was settled seconds or minutes later;
- has the same date;
- has the same business date;
- has the same table;
- has the same staff;
- has the same POS ID;
- appears nearby in an EJ file;
- has a similar amount;
- has similar items;
- has the same or similar tender.

Example:

BILL
Slip: 000001018B000026033
Amount: 4433.39

INVOICE
Slip: 000001018B000026037
Amount: 1258.13

These are DIFFERENT TRANSACTIONS.

DO NOT compare 4433.39 with 1258.13.

DO NOT report Amount Change based on this comparison.

DO NOT report Item Removal based on this comparison.

DO NOT report Item Addition based on this comparison.

DO NOT use the Invoice's tender for Slip 000001018B000026033.

DO NOT use the Invoice's Cash payment to increase the risk tier of
Slip 000001018B000026033.

The Invoice belongs only to Slip 000001018B000026037.

============================================================
MISSING SAME-SLIP DOCUMENT RULE
============================================================

If the transaction being analyzed does not contain an Invoice whose
explicit "Slip:" value exactly matches the PRIMARY SLIP NUMBER:

DO NOT search for another Invoice.

DO NOT use the nearest Invoice.

DO NOT use the next Invoice.

DO NOT use a previous Invoice.

DO NOT use an Invoice with a nearby timestamp.

DO NOT infer which Invoice probably belongs to the transaction.

Treat the Invoice as unavailable.

Return:

invoice_number = ""
final_invoice_amount = null

Apply the same rule to every other missing document or value.

Missing information must remain missing.

NEVER fill missing information using another Slip Number.

============================================================
MULTIPLE EJ FILES
============================================================

These rules apply across ALL supplied EJ files.

Documents from different EJ files may be combined ONLY when their
explicit "Slip:" values are exactly identical.

Same explicit Slip Number across multiple EJ files:
COMBINE.

Different explicit Slip Numbers in the same or different EJ files:
NEVER COMBINE.

Do not use dates, times, file order, document order, or physical
proximity to connect records across EJ files.

Dates and times may be used only to describe or order documents AFTER
those documents have already been matched by exact Slip Number.

============================================================
MULTIPLE SLIPS IN ONE REQUEST
============================================================

Several Slip trails may be supplied in one Gemini request for
processing efficiency.

This does not change transaction ownership.

Every Slip remains a completely independent transaction.

Analyze each PRIMARY SLIP NUMBER separately.

Never carry an Invoice, Bill, amount, item, discount, tender,
transaction number, staff value, date, time, or evidence from one
Slip into another Slip's analysis.

============================================================
FINAL VALIDATION
============================================================

Before returning ANY exception, verify that every document and every
value used as evidence belongs to the exact returned slip_number.

Specifically verify the Slip Number for every:

- Bill
- Print Bill
- Rebill
- Invoice
- Bill Cancellation
- Return
- Refund
- amount
- item
- discount
- tender
- transaction number
- staff value
- evidence entry

If any information came from a document whose explicit "Slip:" value
is different from the returned slip_number:

DISCARD THAT INFORMATION.

Reassess the transaction using only documents whose explicit "Slip:"
value exactly matches the returned slip_number.

If the exception no longer exists after removing information belonging
to another Slip:

DO NOT RETURN THE EXCEPTION.

============================================================
NON-NEGOTIABLE RULE
============================================================

"Slip:" IS THE ONLY TRANSACTION KEY.

Not Date.
Not Time.
Not Settlement Time.
Not Bill Slip No.
Not Bill Slip Ref. No.
Not Transaction Number.
Not Invoice Number.
Not Table.
Not Staff.
Not POS ID.
Not Amount.
Not Items.
Not Tender.
Not document proximity.

ONLY THE EXACT VALUE PRINTED AFTER "Slip:".

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
SCREEN EVERY SUPPLIED SLIP
============================================================

Every supplied PRIMARY SLIP NUMBER must be screened.

The fact that a Slip was supplied to you does NOT mean that the
transaction is suspicious or that an exception already exists.

Python has not pre-classified the supplied Slips as suspicious.

For EACH supplied PRIMARY SLIP NUMBER:

1. Review the complete same-Slip transaction trail.
2. Apply all exception criteria defined in this prompt.
3. Determine whether the evidence actually supports an exception.
4. If one or more criteria are supported, return the Slip as an
   exception.
5. If no criterion is supported, DO NOT return that Slip in the
   exceptions array.

Do not create an exception merely because a transaction was supplied
for analysis.

Normal transactions must not be returned as exceptions.

When none of the supplied Slips meet an exception criterion, return:

{
  "exceptions": []
}

============================================================
IRREGULARITY DETECTION CRITERIA
============================================================

Screen EVERY supplied PRIMARY SLIP NUMBER against ALL seven approved
irregularity detection criteria below.

The fact that a Slip is supplied for analysis does NOT mean that the
transaction is irregular.

Python has not pre-classified the supplied Slip as suspicious.

For every PRIMARY SLIP NUMBER:

1. Review the complete transaction trail belonging to that exact
   Slip Number.

2. Evaluate ALL seven approved irregularity detection criteria
   independently.

3. Use ONLY documents whose explicit "Slip:" value exactly matches
   the current PRIMARY SLIP NUMBER.

4. Do not stop screening after identifying the first exception.

5. A single Slip may satisfy more than one approved criterion.

6. Return the Slip as an exception only when the supplied same-Slip
   evidence supports at least one approved criterion.

7. If none of the seven approved criteria are supported, DO NOT
   return the Slip in the exceptions array.

8. Never use evidence from another Slip Number to establish an
   irregularity.

9. Do not invent missing amounts, items, discounts, tenders,
   timestamps, identification details, or transaction references.

10. When a criterion requires a chronological sequence, establish the
    sequence only from same-Slip evidence.

============================================================
1. MULTIPLE TENDER - BOTH TENDERS HAVE DECIMAL AMOUNTS
============================================================

Finding / Rule Triggered:

Multiple Tender – Both Tenders Have Decimals

SharePoint Flag Category:

Multiple Tender Transaction

Flag only when ALL of the following are supported:

1. The same PRIMARY SLIP NUMBER contains two or more actual payment
   tender entries.

2. At least two tender amounts are explicitly displayed in the
   transaction.

3. At least two tender amounts contain a NON-ZERO decimal portion.

Examples that meet the criterion:

Cash                     100.25
Credit Card               50.75

or:

GCash                  1,250.50
Cash                     500.25

Examples that do NOT meet the criterion:

Cash                     100.00
Credit Card               50.00

or:

Cash                     100.25
Credit Card               50.00

The .00 monetary formatting by itself does not count as a non-zero
decimal amount.

Capture the exact:

- tender description;
- tender amount;
- number of tenders;
- relevant same-Slip payment evidence.

Preserve the tender descriptions and amounts exactly as displayed in
the TXT transaction trail.

Do NOT:

- round tender amounts;
- recompute tender amounts;
- change their displayed values;
- combine tender amounts;
- infer a missing tender amount;
- flag merely because multiple tenders exist;
- flag when fewer than two tender amounts contain non-zero decimal
  portions.

============================================================
2. FULL TRANSACTION VOID
============================================================

Finding / Rule Triggered:

Full Transaction Void

SharePoint Flag Category:

Void Transaction

Flag when same-Slip evidence establishes that the ENTIRE transaction,
entire customer bill, or finalized transaction was voided, cancelled,
or completely reversed.

The void must apply to the complete transaction.

Do NOT classify the following by themselves as a Full Transaction
Void:

- individual item void;
- modifier void;
- one removed item;
- item deletion;
- quantity reduction;
- partial cancellation;
- partial transaction correction.

Look for evidence that specifically supports a complete transaction
void, cancellation, or reversal.

Capture when explicitly available:

- Slip Number;
- transaction number;
- Invoice number;
- original transaction amount;
- void or cancellation amount;
- void description;
- cancellation or reversal reference;
- relevant date and time.

Do NOT flag Full Transaction Void merely because the word "VOID"
appears somewhere in the transaction.

If a void exists but the evidence does not establish that the entire
transaction was voided, do not describe it as a confirmed Full
Transaction Void.

============================================================
3. LAST PRINT BILL - INVOICE TRANSACTION MISMATCH
============================================================

Finding / Rule Triggered:

Last Print Bill – Invoice Transaction Mismatch

SharePoint Flag Category:

Last Bill not Tally Invoice

Identify the LAST customer Print Bill occurring before the finalized
same-Slip Invoice.

The comparison MUST use only the Print Bill and Invoice whose explicit
"Slip:" values match the same PRIMARY SLIP NUMBER.

Compare the last customer Print Bill with the finalized same-Slip
Invoice.

Evaluate material differences including:

- total amount difference;
- item removal;
- item addition;
- item substitution;
- quantity change;
- price change;
- discount addition;
- discount removal;
- discount replacement;
- material charge difference;
- other material customer-payable differences.

Flag when the LAST customer Print Bill materially does not match the
finalized same-Slip Invoice.

Capture when available:

- first Print Bill amount;
- subsequent Print Bill amounts;
- last Print Bill amount;
- Final Invoice Amount;
- amount difference;
- removed items;
- added items;
- changed quantities;
- changed prices;
- changed discounts;
- other supported material differences.

Multiple Print Bills alone are NOT an irregularity.

Do not flag when a customer Print Bill was simply reprinted and the
following remained materially unchanged:

- items;
- quantities;
- prices;
- discounts;
- charges;
- total amount.

Do not treat:

- formatting differences;
- spacing differences;
- headers;
- footers;
- printing differences;
- other non-material presentation differences

as a Print Bill versus Invoice mismatch.

Never compare the customer Print Bill with an Invoice belonging to
another Slip Number.

If a same-Slip Invoice does not exist, do not use a different Slip's
Invoice to perform the comparison.

============================================================
4. MULTIPLE DISCOUNT APPLICATIONS
============================================================

Finding / Rule Triggered:

Multiple Discount Applications

SharePoint Flag Category:

Multiple Discount

Identify two or more discount applications within the same PRIMARY
SLIP NUMBER.

Capture when explicitly available:

- discount name;
- discount description;
- discount code;
- discount percentage;
- discount amount;
- item affected;
- transaction amount;
- customer payable amount;
- VAT-exempt treatment;
- related transaction details.

Determine whether the multiple discounts were applied to the same
transaction or customer payable amount.

Differentiate multiple transaction-level discounts from separate
legitimate item-level discounts whenever the EJ provides enough
information.

Flag when multiple discount applications are present on the same
transaction and require review.

Do not flag a transaction merely because one discount exists.

Do not automatically conclude that multiple discounts are
unauthorized.

If multiple discounts clearly exist but the transaction trail does not
provide enough information to determine whether the combination is
legitimate or overlapping, identify the transaction for human review
and clearly state that the discount combination requires verification.

============================================================
5. LONG OUTSTANDING TRANSACTION - 5 HOURS OR MORE
============================================================

Finding / Rule Triggered:

Long Outstanding Transaction – ≥5 Hours

SharePoint Flag Category:

Long Outstanding Transaction

Flag when the SAME Slip transaction trail establishes an outstanding
transaction duration of at least:

5 hours
or
300 minutes.

Identify a supported starting event and ending event belonging to the
exact same PRIMARY SLIP NUMBER.

Possible supported start events may include:

- transaction opening;
- transaction initiation;
- first relevant same-Slip transaction activity;
- suspension;
- hold;
- pending status;
- customer Print Bill when it represents the supported beginning of
  the outstanding period.

Possible supported ending events may include:

- final Invoice;
- payment completion;
- settlement;
- transaction closure;
- cancellation;
- latest supported same-Slip event when the transaction is explicitly
  still open, suspended, pending, or unfinished.

Calculate:

Elapsed Duration = Supported End Timestamp - Supported Start Timestamp

Flag only when:

Elapsed Duration >= 5 hours.

Capture:

- start timestamp;
- end timestamp;
- duration in hours and minutes;
- events used to establish the duration;
- whether the transaction was completed or remained outstanding.

Example:

Start: 10:15 AM
End: 3:45 PM

Duration:

5 hours 30 minutes

This meets the criterion.

Date and time may be used for this duration calculation ONLY AFTER the
records have already been associated using the exact same "Slip:"
value.

Date and time must NEVER be used to decide which transaction a
document belongs to.

Do NOT:

- use timestamps belonging to another Slip Number;
- connect unrelated EJ timestamps;
- use physical proximity as proof that timestamps belong together;
- use document position as evidence of duration;
- invent a missing start timestamp;
- invent a missing end timestamp;
- calculate duration when the transaction sequence cannot be
  established reliably.

============================================================
6. POST-SETTLEMENT TRANSACTION CHANGE
============================================================

Finding / Rule Triggered:

Post-Settlement Transaction Change

SharePoint Flag Category:

Post - Settlement Change

First establish that the SAME Slip transaction was settled, finalized,
paid, or closed.

Then determine whether a material transaction change occurred AFTER
that supported settlement or finalization event.

Potential post-settlement modifications include:

- item removal;
- item addition;
- item substitution;
- quantity change;
- price change;
- discount addition;
- discount removal;
- discount replacement;
- tender change;
- payment reversal;
- cancellation;
- rebill;
- material correction;
- other material transaction modification.

Flag only when the transaction trail supports the chronological
sequence:

SETTLEMENT OR FINALIZATION
followed by
MATERIAL TRANSACTION CHANGE.

Capture when available:

- original transaction details;
- settlement or finalization event;
- settlement timestamp;
- subsequent modification;
- modification timestamp;
- original amount;
- changed amount;
- affected items;
- discount change;
- tender change;
- transaction references.

Do not flag modifications that occurred before settlement.

Do not assume that an EJ entry occurred after settlement merely because
the entry appears later in the TXT file.

Physical document position alone is NOT evidence of a post-settlement
change.

The sequence must be supported by evidence within the SAME PRIMARY
SLIP NUMBER.

Date and time may be used to establish chronology only after
transaction ownership has already been established by matching the
exact "Slip:" value.

============================================================
7. VAT-EXEMPT - MISSING OSCA/PWD/SENIOR DETAILS
============================================================

Finding / Rule Triggered:

VAT-Exempt Transaction – Missing OSCA/PWD/Senior Details

SharePoint Flag Category:

OSCA/PWD/ Senior Details

First determine whether the same-Slip transaction is explicitly
identified as:

- Senior Citizen;
- OSCA;
- PWD;
- Person With Disability;
- Senior Citizen discount;
- PWD discount;
- Senior Citizen VAT exemption;
- PWD VAT exemption;
- or another explicitly identified Senior Citizen or PWD-related
  VAT-exempt transaction.

Then identify the identification details applicable to that
transaction.

For a Senior Citizen / OSCA transaction, check for applicable
Senior Citizen or OSCA identification details, including:

- OSCA/Senior Citizen ID; and
- Senior Citizen customer name.

For a PWD transaction, check for applicable PWD identification
details, including:

- PWD ID; and
- PWD customer name.

Flag when the applicable required identification evidence is missing,
including when:

- the applicable ID is missing;
- the applicable customer name is missing;
- both the applicable ID and customer name are missing.

Clearly state exactly which detail is missing.

Examples:

Senior Citizen transaction:
OSCA ID available
Senior Citizen Name missing
→ Flag.

Senior Citizen transaction:
OSCA ID missing
Senior Citizen Name available
→ Flag.

PWD transaction:
PWD ID available
PWD Name missing
→ Flag.

PWD transaction:
PWD ID missing
PWD Name available
→ Flag.

Capture when explicitly available:

- Slip Number;
- transaction number;
- Invoice number;
- transaction amount;
- VAT-exempt amount;
- discount description;
- applicable OSCA/Senior/PWD ID;
- applicable customer name;
- relevant date and time.

Do NOT invent missing identification information.

A blank field, empty field, empty label, or identification label with
no corresponding value must be treated as missing.

Do not assume that every VAT-exempt transaction is automatically a
Senior Citizen or PWD transaction.

If another VAT-exemption category is explicitly identified and there
is no evidence that the transaction is Senior Citizen, OSCA, or PWD
related, do not require OSCA/PWD/Senior identification solely because
VAT-exempt information exists.

============================================================
FINAL APPROVED-CRITERIA VALIDATION
============================================================

Before returning ANY Slip as an exception:

1. Confirm that the Slip satisfies at least one of the seven approved
   irregularity detection criteria.

2. Confirm that all supporting documents belong to the exact returned
   slip_number.

3. Confirm that all transaction ownership was established using the
   explicit "Slip:" value only.

4. Confirm that no information was imported from another Slip Number.

5. Preserve exact displayed tender amounts and other critical evidence
   whenever describing findings.

6. Do not invent missing amounts, timestamps, tenders, discounts,
   OSCA/PWD/Senior details, transaction references, or documents.

7. Do not return a Slip merely because it was supplied to Gemini.

8. If multiple approved criteria independently apply to the same Slip,
   preserve all applicable flag_categories and describe all supported
   findings.

9. If none of the seven approved criteria are supported, DO NOT return
   the Slip in the exceptions array.

============================================================
DO NOT FLAG SOLELY BECAUSE
============================================================

The presence of an event, document, amount, discount, tender, timestamp,
or other transaction characteristic does NOT by itself establish an
approved irregularity.

Do NOT flag a Slip solely because:

- the same Slip Number appears repeatedly;

- the same Slip Number appears in more than one EJ file;

- production-printer records repeat;

- BAR, WINDOW, NACHO, KITCHEN, MEAT, MIDDLE, PANTRY, PASTRY,
  GRILL, FRYER, DESSERT, BEVERAGE, or SERVICE BAR records repeat;

- an Invoice exists;

- a customer Print Bill exists;

- multiple customer Print Bills exist;

- a customer Print Bill was reprinted without any material transaction
  change;

- an unchanged customer Print Bill appears multiple times;

- a transaction crosses midnight;

- one discount exists;

- VAT-exempt treatment exists;

- Senior Citizen, OSCA, or PWD information exists;

- a transaction contains multiple payment tenders;

- one tender amount contains a non-zero decimal portion;

- multiple tender amounts are displayed with .00 monetary formatting;

- the word VOID appears somewhere in the transaction;

- an individual item was voided;

- an individual modifier was voided;

- one item was removed or deleted;

- a partial transaction correction occurred;

- a transaction contains timestamps that are five or more hours apart
  unless those timestamps establish the supported outstanding period
  of the exact same Slip transaction;

- an EJ record appears after an Invoice, settlement, or payment record
  merely because of its physical location in the TXT file;

- an event appears later in an EJ file without supported evidence that
  the event actually occurred after settlement;

- a negative amount appears;

- a transaction contains missing information that is unrelated to one
  of the seven approved criteria;

- a transaction looks unusual but does not satisfy at least one of the
  seven approved criteria.

============================================================
CRITERION-SPECIFIC FALSE-POSITIVE CONTROLS
============================================================

1. MULTIPLE TENDER

Do NOT flag Multiple Tender Transaction merely because two or more
tenders exist.

The approved Multiple Tender criterion requires at least two actual
tender amounts with NON-ZERO decimal portions.

Example:

Cash             100.25
Credit Card       50.00

Do NOT flag under this criterion because only one tender amount has a
non-zero decimal portion.


2. FULL TRANSACTION VOID

Do NOT flag Void Transaction merely because VOID appears in the trail.

An item void, modifier void, item deletion, or partial cancellation is
not by itself a Full Transaction Void.

The evidence must establish that the entire transaction was voided,
cancelled, or completely reversed.


3. LAST PRINT BILL - INVOICE TRANSACTION MISMATCH

Do NOT flag Last Bill not Tally Invoice merely because:

- multiple Print Bills exist;
- a Bill was reprinted;
- document formatting changed;
- printing details changed;
- headers or footers differ.

There must be a supported material difference between the LAST customer
Print Bill and the finalized Invoice belonging to the exact same Slip.


4. MULTIPLE DISCOUNT

Do NOT flag Multiple Discount merely because one discount exists.

Do NOT automatically treat separate legitimate item-level discounts as
multiple transaction-level discount applications.

There must be evidence of two or more discount applications relevant
to the same transaction or payable amount, or sufficient evidence that
the combination requires human verification.


5. LONG OUTSTANDING TRANSACTION

Do NOT flag Long Outstanding Transaction merely because two same-Slip
timestamps are five or more hours apart.

The timestamps must establish the supported start and end of an
outstanding, open, pending, suspended, or unfinished transaction
period.

The calculated duration must be at least 5 hours or 300 minutes.

Never use timestamps from another Slip Number.


6. POST-SETTLEMENT TRANSACTION CHANGE

Do NOT flag Post - Settlement Change merely because one event appears
after another event in the TXT file.

First establish settlement, payment completion, finalization, or
transaction closure.

Then establish that a material same-Slip modification actually occurred
after that event.

Physical document position alone is not evidence of post-settlement
activity.


7. OSCA/PWD/SENIOR DETAILS

Do NOT flag OSCA/PWD/ Senior Details merely because:

- VAT-exempt information exists;
- a discount exists;
- Senior Citizen, OSCA, or PWD information exists.

First establish that the same-Slip transaction is explicitly a Senior
Citizen, OSCA, or PWD-related transaction for which the applicable
identification details are required.

Then determine whether the applicable ID, customer name, or both are
actually missing.

Do not apply OSCA requirements to another explicitly identified
VAT-exemption category merely because the transaction is VAT-exempt.

============================================================
FINAL FALSE-POSITIVE CHECK
============================================================

Before returning a Slip:

1. Identify the exact approved criterion that was satisfied.

2. Confirm that the evidence satisfies the specific conditions of that
   criterion.

3. Confirm that the finding is not based solely on one of the excluded
   circumstances above.

4. Confirm that all evidence belongs to the exact same PRIMARY SLIP
   NUMBER.

5. If the transaction does not satisfy at least one approved criterion,
   DO NOT return it as an exception.

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

For flag_categories, use ONLY the exact SharePoint Choice values
specified below.

The allowed Flag Category values for the seven approved criteria are:

1. Multiple Tender – Both Tenders Have Decimals

   flag_categories:
   ["Multiple Tender Transaction"]

2. Full Transaction Void

   flag_categories:
   ["Void Transaction"]

3. Last Print Bill – Invoice Transaction Mismatch

   flag_categories:
   ["Last Bill not Tally Invoice"]

4. Multiple Discount Applications

   flag_categories:
   ["Multiple Discount"]

5. Long Outstanding Transaction – ≥5 Hours

   flag_categories:
   ["Long Outstanding Transaction"]

6. Post-Settlement Transaction Change

   flag_categories:
   ["Post - Settlement Change"]

7. VAT-Exempt Transaction – Missing OSCA/PWD/Senior Details

   flag_categories:
   ["OSCA/PWD/ Senior Details"]

The value returned in flag_categories must match the configured
SharePoint Choice value exactly.

Preserve the exact:

- spelling;
- capitalization;
- spacing;
- punctuation;
- slash characters;
- hyphen placement.

Do not create a new Flag Category value.

Do not output the detailed finding label as the Flag Category when its
corresponding SharePoint Choice value is different.

Examples:

Correct:

rule_triggered =
"Multiple Tender – Both Tenders Have Decimals"

flag_categories =
["Multiple Tender Transaction"]


Correct:

rule_triggered =
"Full Transaction Void"

flag_categories =
["Void Transaction"]


Correct:

rule_triggered =
"VAT-Exempt Transaction – Missing OSCA/PWD/Senior Details"

flag_categories =
["OSCA/PWD/ Senior Details"]


Incorrect:

flag_categories =
["Multiple Tender – Both Tenders Have Decimals"]


Incorrect:

flag_categories =
["Full Transaction Void"]


Incorrect:

flag_categories =
["VAT-Exempt Transaction – Missing OSCA/PWD/Senior Details"]


The detailed irregularity label must be returned in rule_triggered.

The corresponding exact SharePoint Choice value must be returned in
flag_categories.

When one Slip independently satisfies more than one approved
criterion, return every applicable SharePoint Choice value in
flag_categories.

Example:

flag_categories = [
  "Multiple Tender Transaction",
  "OSCA/PWD/ Senior Details"
]

rule_triggered =
"Multiple Tender – Both Tenders Have Decimals; "
"VAT-Exempt Transaction – Missing OSCA/PWD/Senior Details"

Do not return any category solely because the category exists in
SharePoint.

Return a category only when the same-Slip evidence satisfies its
corresponding approved irregularity criterion.

Do not use Return/Refund as a Flag Category because Return / Refund
Transaction is not included in the current seven approved irregularity
detection criteria.

Do not use Unusual Transaction Sequence as a fallback category.

Do not create or return an exception that does not satisfy at least
one of the seven approved criteria.
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


Evidence must explain WHY the transaction satisfies one or more of the
seven approved irregularity detection criteria.

For every returned exception:

1. Identify the exact approved criterion or criteria triggered.

2. Populate rule_triggered using the detailed approved finding label.

3. Populate flag_categories using only the corresponding exact
   SharePoint Choice value.

4. Explain the transaction-specific irregularity clearly in finding.

5. Preserve the relevant transaction sequence in evidence_trail.

6. Use only information from documents whose explicit "Slip:" value
   exactly matches the returned slip_number.

============================================================
EXACT VALUE PRESERVATION
============================================================

When describing evidence, preserve critical values exactly as displayed
in the supplied TXT transaction trail.

This applies especially to:

- tender descriptions;
- tender amounts;
- Print Bill amounts;
- Invoice amounts;
- discount descriptions;
- discount amounts;
- item descriptions;
- quantities;
- transaction numbers;
- Invoice numbers;
- dates;
- times;
- OSCA IDs;
- Senior Citizen IDs;
- PWD IDs;
- customer names;
- void-related references.

Do NOT:

- round displayed values;
- recompute displayed amounts;
- change displayed decimal portions;
- remove a displayed negative sign;
- add a negative sign that was not displayed;
- combine separate amounts;
- infer a missing amount;
- infer a missing ID;
- infer a missing customer name;
- invent a missing timestamp;
- invent a transaction reference.

============================================================
MULTIPLE TENDER EVIDENCE
============================================================

For:

Multiple Tender – Both Tenders Have Decimals

evidence_trail must identify each relevant tender separately.

Example:

Tender 1: Cash = 100.25
Tender 2: Credit Card = 50.75

Preserve the exact tender descriptions and displayed amounts.

Do not merely state:

"Multiple tender found."

Show which tenders and amounts caused the criterion to be satisfied.

============================================================
FULL TRANSACTION VOID EVIDENCE
============================================================

For:

Full Transaction Void

evidence_trail must identify the evidence establishing that the ENTIRE
transaction was voided, cancelled, or completely reversed.

When available, include:

- original transaction amount;
- void/cancellation amount;
- transaction number;
- Invoice number;
- void/cancellation reference;
- supporting transaction event;
- relevant date and time.

Do not describe an item-level or partial void as evidence of a Full
Transaction Void.

============================================================
LAST PRINT BILL VS INVOICE EVIDENCE
============================================================

For:

Last Print Bill – Invoice Transaction Mismatch

evidence_trail must identify:

- the LAST customer Print Bill;
- the finalized same-Slip Invoice;
- the specific material difference.

Where available, show:

Last Print Bill Amount: <exact displayed amount>
Final Invoice Amount: <exact displayed amount>

Then identify the supported difference, such as:

- amount difference;
- item removal;
- item addition;
- quantity change;
- price change;
- discount change;
- charge difference.

Do not compare records from different Slip Numbers.

============================================================
MULTIPLE DISCOUNT EVIDENCE
============================================================

For:

Multiple Discount Applications

evidence_trail must identify the individual discount applications that
caused the criterion to be satisfied.

Where available, preserve:

- discount name;
- discount description;
- discount amount;
- discount percentage;
- affected item or transaction;
- resulting transaction amount.

Do not merely state:

"Multiple discounts found."

Identify the actual discounts shown in the same-Slip trail.

============================================================
LONG OUTSTANDING TRANSACTION EVIDENCE
============================================================

For:

Long Outstanding Transaction – ≥5 Hours

evidence_trail must identify:

- supported start event;
- exact start timestamp;
- supported end event;
- exact end timestamp;
- calculated elapsed duration.

State the calculated duration in hours and minutes.

Example:

Start Event: Print Bill
Start Timestamp: 10:15 AM
End Event: Final Invoice
End Timestamp: 3:45 PM
Elapsed Duration: 5 hours 30 minutes

Use timestamps only after both events have already been associated
with the exact same PRIMARY SLIP NUMBER.

Never calculate elapsed time using evidence from different Slip
Numbers.

============================================================
POST-SETTLEMENT CHANGE EVIDENCE
============================================================

For:

Post-Settlement Transaction Change

evidence_trail must establish both:

1. the supported settlement/finalization event; and
2. the subsequent material change.

When available, preserve:

- settlement event;
- settlement timestamp;
- original transaction state;
- subsequent modification;
- modification timestamp;
- affected amount;
- affected item;
- affected discount;
- affected tender.

The evidence must establish the chronological sequence.

Do not use physical TXT file position alone as proof that a change
occurred after settlement.

============================================================
OSCA/PWD/SENIOR DETAILS EVIDENCE
============================================================

For:

VAT-Exempt Transaction – Missing OSCA/PWD/Senior Details

first identify whether the supported transaction is:

- Senior Citizen / OSCA-related; or
- PWD-related.

Then explicitly state which required information is available and
which information is missing.

Examples:

Senior Citizen / OSCA:
ID: Available
Name: Missing

or:

PWD:
ID: Missing
Name: Available

or:

PWD:
ID: Missing
Name: Missing

When an ID or name is available, preserve the actual displayed value
in evidence_trail.

When an ID or name is missing, state:

Missing

Do not invent the missing value.

============================================================
INCOMPLETE OR UNCERTAIN EVIDENCE
============================================================

Do not transform incomplete evidence into a confirmed fact.

If an event is visible but the information necessary to satisfy a
criterion is unavailable, explain exactly what is unavailable.

For example:

- void exists but full-transaction scope is not established;
- timestamps exist but outstanding duration cannot reliably be
  established;
- discounts exist but their applicability cannot be determined;
- transaction modification exists but the evidence does not establish
  that it occurred after settlement.

Where the corresponding approved criterion specifically permits human
verification of incomplete evidence, clearly state:

Requires verification.

Otherwise, do not return the Slip as satisfying that criterion.

============================================================
MISSING VALUES
============================================================

If a schema field cannot be reliably populated from the same-Slip
evidence:

- use an empty string for unavailable text fields;
- use null for unavailable optional numeric fields;
- use 0 only where the numeric field legitimately represents a count
  of zero.

Never obtain a missing value from another Slip Number.

============================================================
FINAL EVIDENCE CHECK
============================================================

Before returning each exception, verify:

1. Every material fact in finding is supported by the supplied
   same-Slip trail.

2. Every material fact in evidence_trail is supported by the supplied
   same-Slip trail.

3. The rule_triggered corresponds to an approved criterion actually
   satisfied by the evidence.

4. Every value in flag_categories corresponds to an approved criterion
   actually satisfied by the evidence.

5. No evidence belongs to another Slip Number.

6. No missing value was invented.

7. Exact displayed values were preserved where required.

If removing unsupported information causes the approved criterion to
no longer be satisfied, DO NOT return that criterion.

============================================================
RISK TIER
============================================================

Every returned exception must use exactly one Risk Tier:

High
Medium
Low

Assign a Risk Tier only AFTER the transaction satisfies at least one
of the seven approved irregularity detection criteria.

Use only evidence belonging to the exact same PRIMARY SLIP NUMBER.

If multiple findings apply to the same Slip, use the HIGHEST applicable
Risk Tier:

High > Medium > Low

============================================================
HIGH RISK
============================================================

Classify as High ONLY when at least one of the following three High
Risk conditions is supported.

------------------------------------------------------------
HIGH RULE 1:
MULTIPLE TENDER + BOTH TENDERS HAVE DECIMALS + CASH
------------------------------------------------------------

Classify as High when ALL conditions below are supported:

1. The transaction satisfies:

   Multiple Tender – Both Tenders Have Decimals

2. Two or more actual payment tenders exist.

3. At least two tender amounts contain NON-ZERO decimal portions.

4. At least one of the qualifying decimal tenders is Cash.

Example:

Cash                     100.25
Credit Card               50.75

Risk Tier:

High

Another example:

Cash                     500.50
GCash                    250.25

Risk Tier:

High

Do NOT classify as High under this rule when:

Cash                     100.00
Credit Card               50.75

because at least two tender amounts do not contain non-zero decimal
portions.

Cash alone does NOT trigger High.

------------------------------------------------------------
HIGH RULE 2:
LAST PRINT BILL - INVOICE MISMATCH + CASH
------------------------------------------------------------

Classify as High when ALL conditions below are supported:

1. The transaction satisfies:

   Last Print Bill – Invoice Transaction Mismatch

2. The LAST customer Print Bill materially differs from the finalized
   SAME-SLIP Invoice.

3. The difference is supported by one or more material changes,
   including:

   - amount difference;
   - item removal;
   - item addition;
   - item substitution;
   - quantity change;
   - price change;
   - discount difference;
   - material charge difference;
   - another supported material customer-payable difference.

4. The FINAL tender from the finalized SAME-SLIP Invoice is Cash.

Example:

Last Print Bill Amount: 3,000.00
Final Invoice Amount:   2,500.00
Final Tender: Cash

Risk Tier:

High

Cash alone is NOT sufficient.

If the Last Print Bill materially matches the finalized Invoice,
Cash must NOT trigger this High rule.

------------------------------------------------------------
HIGH RULE 3:
VAT EXEMPT + PWD/SENIOR/OSCA DISCOUNT + MISSING DETAILS
------------------------------------------------------------

Classify as High when ALL conditions below are supported:

1. The transaction is explicitly identified as VAT-exempt.

2. The same transaction contains a supported:

   - Senior Citizen / OSCA discount; or
   - PWD discount.

3. Applicable required identification information is missing.

For Senior Citizen / OSCA transactions, missing information may
include:

- OSCA / Senior Citizen ID;
- Senior Citizen customer name;
- both ID and customer name.

For PWD transactions, missing information may include:

- PWD ID;
- PWD customer name;
- both ID and customer name.

Example:

VAT Exempt: Yes
Discount: Senior Citizen
OSCA ID: Missing
Senior Citizen Name: Available

Risk Tier:

High

Example:

VAT Exempt: Yes
Discount: PWD
PWD ID: Available
PWD Name: Missing

Risk Tier:

High

VAT exemption alone must NOT trigger High.

PWD or Senior Citizen discount alone must NOT trigger High.

The missing applicable identification details must also be supported.

============================================================
MEDIUM RISK
============================================================

Classify as Medium when:

1. The approved Full Transaction Void criterion is supported; and

2. None of the High Risk conditions above are independently supported
   by the same Slip.

Approved finding:

Full Transaction Void

SharePoint Flag Category:

Void Transaction

Risk Tier:

Medium

The evidence must establish a complete transaction void.

An individual item void, item deletion, modifier void, or partial
cancellation is NOT automatically a Full Transaction Void.

============================================================
LOW RISK
============================================================

Classify all OTHER supported approved irregularity findings as Low when
the same Slip does not satisfy a High or Medium condition.

This includes:

1. Multiple Tender – Both Tenders Have Decimals

   Risk Tier = Low when the criterion is satisfied but the specific
   High condition involving a qualifying Cash tender is NOT satisfied.

Example:

GCash                    100.25
Credit Card               50.75

Risk Tier:

Low


2. Last Print Bill – Invoice Transaction Mismatch

   Risk Tier = Low when the mismatch is supported but the finalized
   SAME-SLIP Invoice does NOT have Cash as its final tender.

Example:

Last Print Bill: 3,000.00
Final Invoice:   2,500.00
Final Tender: PayMaya

Risk Tier:

Low


3. Multiple Discount Applications

Risk Tier:

Low


4. Long Outstanding Transaction – ≥5 Hours

Risk Tier:

Low


5. Post-Settlement Transaction Change

Risk Tier:

Low


6. OSCA/PWD/Senior Details findings that satisfy the approved
irregularity criterion but do NOT contain all evidence required for
the specific High Risk rule.

Risk Tier:

Low

============================================================
MULTIPLE FINDINGS
============================================================

A single PRIMARY SLIP NUMBER may satisfy more than one approved
irregularity criterion.

When this happens, determine the Risk Tier for each supported finding
and use the highest applicable Risk Tier as the final transaction Risk
Tier.

Priority:

High > Medium > Low

Example:

Multiple Discount Applications
= Low

AND

Full Transaction Void
= Medium

Final Risk Tier:

Medium


Example:

Full Transaction Void
= Medium

AND

Multiple Tender – Both Tenders Have Decimals
with a qualifying Cash tender
= High

Final Risk Tier:

High

============================================================
FINAL RISK-TIER VALIDATION
============================================================

Before returning the final Risk Tier:

1. Confirm that the transaction satisfies at least one of the seven
   approved irregularity detection criteria.

2. Use evidence belonging only to the exact same PRIMARY SLIP NUMBER.

3. Never use an Invoice, tender, amount, discount, VAT treatment,
   identification detail, or other evidence belonging to another Slip.

4. Apply High ONLY when one or more defined High Risk conditions are
   completely supported.

5. Apply Medium for Full Transaction Void when no High Risk condition
   applies.

6. Apply Low for the other supported approved findings when neither
   High nor Medium applies.

7. If multiple supported findings have different Risk Tiers, use the
   highest applicable Risk Tier:

   High > Medium > Low

8. Cash alone does not create High Risk.

9. Multiple tenders alone do not create High Risk.

10. VAT exemption alone does not create High Risk.

11. A discount alone does not create High Risk.

12. A Print Bill alone does not create High Risk.

13. Do not return a normal transaction merely to assign it Low Risk.

Return only Slips satisfying at least one of the seven approved
irregularity detection criteria.

If none of the supplied Slips satisfy an approved criterion, return:

{
  "exceptions": []
}
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
) -> Tuple[EJAnalysis, float, Dict]:

    slip_numbers = [
        normalize_slip_number(
            slip_number
        )
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

STRICT TRANSACTION RULES:

1. THE VALUE PRINTED AFTER "Slip:" IS THE ONLY TRANSACTION KEY.

2. For every document, first locate its explicit:

   Slip: <value>

   The value after "Slip:" determines which transaction owns the
   entire document.

3. A document belongs to the current PRIMARY SLIP NUMBER ONLY when:

   document "Slip:" value == PRIMARY SLIP NUMBER

4. If a document's "Slip:" value is different from the PRIMARY SLIP
   NUMBER, completely ignore that document for the current
   transaction.

5. NEVER use any of the following to associate documents:

   - Date
   - Business Date
   - Transaction Date
   - Printed Date
   - Time
   - Transaction Time
   - Printed Time
   - Settlement Time
   - nearby or consecutive timestamps
   - Bill Slip No.
   - Bill Slip Ref. No.
   - Invoice Number
   - Transaction Number
   - Table Number
   - Staff
   - POS ID
   - Guest Number
   - Amount
   - Tender
   - Items
   - Discounts
   - document order
   - document proximity
   - EJ file order
   - EJ file name

6. DATE AND TIME ARE NOT TRANSACTION KEYS.

   Never associate documents because they occurred seconds or minutes
   apart.

   Never associate documents because their settlement times are close.

   Never associate documents because they occurred on the same date.

7. Dates and times may be used only for description or chronological
   ordering AFTER the documents have already been matched using the
   exact same "Slip:" value.

8. KOT, WINDOW, PASTRY, PANTRY, BAR, NACHO, KITCHEN, BILL,
   PRINT BILL, REBILL, BILL CANCELLATION, INVOICE, RETURN, and REFUND
   may form one transaction trail ONLY when each document's explicit
   "Slip:" value exactly matches the PRIMARY SLIP NUMBER.

9. An INVOICE belongs to the current transaction ONLY when the
   Invoice document explicitly contains:

   Slip: <PRIMARY SLIP NUMBER>

   Never use the nearest Invoice.
   Never use the next Invoice.
   Never use the previous Invoice.
   Never use an Invoice because its transaction or settlement time
   is close to the current Bill.

10. Example:

    PRIMARY SLIP NUMBER:
    000001018B000026033

    BILL:
    Slip: 000001018B000026033
    Amount: 4433.39

    INVOICE:
    Slip: 000001018B000026037
    Amount: 1258.13
    Tender: Cash

    The Invoice MUST NOT be used for
    Slip 000001018B000026033.

    DO NOT compare 4433.39 and 1258.13.
    DO NOT report Amount Change.
    DO NOT report Item Removal.
    DO NOT report Item Addition.
    DO NOT use the Cash tender.
    DO NOT classify Slip 000001018B000026033 as High based on this
    Invoice.

11. If an Invoice with the exact same "Slip:" value does not exist
    inside the current PRIMARY SLIP NUMBER boundary, treat the Invoice
    as unavailable.

    Return:

    invoice_number = ""
    final_invoice_amount = null

    Do not obtain the missing Invoice from another Slip.

12. The same rule applies to all extracted values.

    Never import from another Slip:

    - transaction_number
    - bill_slip_number
    - invoice_number
    - return_slip_number
    - pos_id
    - staff
    - business_date
    - bill amounts
    - invoice amount
    - items removed
    - items added
    - discounts
    - tenders
    - evidence

13. Several PRIMARY SLIP NUMBER trails may appear in this Gemini
    request because multiple transactions are processed in one batch.

    This is only for processing efficiency.

    Every PRIMARY SLIP NUMBER remains an independent transaction.

14. Analyze one PRIMARY SLIP NUMBER boundary at a time.

    Use only information between:

    PRIMARY SLIP NUMBER: <current Slip>

    and:

    END PRIMARY SLIP NUMBER: <current Slip>

15. Never carry any extracted value from the previous Slip into the
    next Slip.

16. Before analyzing the next PRIMARY SLIP NUMBER, reset all
    transaction-specific values.

17. Same-Slip records from different SOURCE EJ FILES may be combined
    ONLY when their explicit "Slip:" values are exactly identical.

18. Different Slip Numbers must NEVER be combined, even when the
    documents have:

    - consecutive timestamps;
    - the same settlement period;
    - the same table;
    - the same staff;
    - the same POS ID;
    - similar items;
    - similar amounts;
    - the same tender;
    - adjacent positions in an EJ file.

19. Copy slip_number exactly from the current PRIMARY SLIP NUMBER.

20. Before returning each exception, verify that EVERY populated value
    came from documents whose explicit "Slip:" value exactly matches
    the returned slip_number.

21. If ANY populated value came from a document with a different
    "Slip:" value:

    - discard that value;
    - remove that different-Slip document from consideration;
    - reassess the current Slip using only same-Slip documents.

22. If the proposed exception disappears after removing information
    belonging to another Slip, DO NOT RETURN THAT EXCEPTION.

23. If information is absent from the current same-Slip trail, return
    an empty string or null.

    NEVER fill missing information using another Slip.

FINAL TRANSACTION-IDENTITY CHECK:

"Slip:" IS THE ONLY TRANSACTION KEY.

NOT Date.
NOT Time.
NOT Settlement Time.
NOT Bill Slip No.
NOT Bill Slip Ref. No.
NOT Invoice Number.
NOT Transaction Number.
NOT Table.
NOT Staff.
NOT POS ID.
NOT Amount.
NOT Items.
NOT Tender.
NOT document proximity.

ONLY THE EXACT VALUE AFTER "Slip:".

TRANSACTION TRAILS START
============================================================
{combined_trails}
============================================================
TRANSACTION TRAILS END

Analyze every supplied Slip independently.

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

    # --------------------------------------------------------
    # GEMINI TOKEN USAGE
    # --------------------------------------------------------

    usage_metadata = getattr(
        response,
        "usage_metadata",
        None,
    )

    token_usage = {
        "prompt_tokens": (
            getattr(
                usage_metadata,
                "prompt_token_count",
                0,
            )
            or 0
        ),
        "output_tokens": (
            getattr(
                usage_metadata,
                "candidates_token_count",
                0,
            )
            or 0
        ),
        "total_tokens": (
            getattr(
                usage_metadata,
                "total_token_count",
                0,
            )
            or 0
        ),
    }

    logger.info(
        (
            "Gemini batch %s token usage: "
            "prompt=%s output=%s total=%s"
        ),
        batch_number,
        token_usage["prompt_tokens"],
        token_usage["output_tokens"],
        token_usage["total_tokens"],
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

    allowed_slips = set(
        slip_numbers
    )

    validated_exceptions: List[
        POSException
    ] = []

    for exception in result.exceptions:
        returned_slip = (
            normalize_slip_number(
                exception.slip_number
            )
        )

        if returned_slip not in allowed_slips:
            logger.error(
                (
                    "Gemini exception rejected. "
                    "Returned Slip=%s; "
                    "authorized Slip Numbers=%s."
                ),
                returned_slip,
                sorted(
                    allowed_slips
                ),
            )

            continue

        exception.slip_number = (
            returned_slip
        )

        validated_exceptions.append(
            exception
        )

    result.exceptions = (
        validated_exceptions
    )

    return (
        result,
        elapsed,
        token_usage,
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
# DETERMINISTIC RISK-TIER RULES
# ============================================================

def enforce_risk_tier_rules(
    exception: POSException,
) -> POSException:
    """
    Final deterministic Risk Tier control.

    HIGH:
    1. Multiple Tender - Both Tenders Have Decimals
       + qualifying Cash tender
    2. Last Print Bill - Invoice Mismatch
       + final same-Slip Cash tender
    3. VAT Exempt + PWD/Senior/OSCA discount
       + missing required details

    MEDIUM:
    - Full Transaction Void

    LOW:
    - Other approved findings

    Priority:
    High > Medium > Low
    """

    rule_triggered = str(
        exception.rule_triggered or ""
    ).strip().upper()

    flag_categories = {
        str(category or "").strip().upper()
        for category in exception.flag_categories
        if category
    }

    # --------------------------------------------------------
    # 1. DEFAULT APPROVED FINDING = LOW
    # --------------------------------------------------------

    final_risk_tier = "Low"

    # --------------------------------------------------------
    # 2. FULL TRANSACTION VOID = MEDIUM
    # --------------------------------------------------------

    has_full_transaction_void = (
        "FULL TRANSACTION VOID"
        in rule_triggered
        or "VOID TRANSACTION"
        in flag_categories
    )

    if has_full_transaction_void:
        final_risk_tier = "Medium"

    # --------------------------------------------------------
    # 3. MULTIPLE TENDER HIGH RULE
    #
    # Gemini has already evaluated:
    # - multiple actual tenders
    # - at least two non-zero decimal tender amounts
    # - one qualifying tender is Cash
    #
    # If Gemini classified the approved finding as High,
    # preserve High.
    # --------------------------------------------------------

    has_multiple_tender_rule = (
        "MULTIPLE TENDER"
        in rule_triggered
        or "MULTIPLE TENDER TRANSACTION"
        in flag_categories
    )

    gemini_marked_high = (
        str(
            exception.risk_tier or ""
        ).strip().upper()
        == "HIGH"
    )

    if (
        has_multiple_tender_rule
        and gemini_marked_high
    ):
        final_risk_tier = "High"

    # --------------------------------------------------------
    # 4. VAT EXEMPT + OSCA/PWD/SENIOR HIGH RULE
    #
    # Gemini evaluates whether:
    # - VAT exemption exists
    # - applicable PWD/Senior/OSCA discount exists
    # - applicable identification detail is missing
    #
    # Preserve High only when Gemini returned High for the
    # approved OSCA/PWD/Senior criterion.
    # --------------------------------------------------------

    has_osca_pwd_senior_rule = (
        "OSCA/PWD/SENIOR"
        in rule_triggered
        or "OSCA/PWD/ SENIOR DETAILS"
        in flag_categories
    )

    if (
        has_osca_pwd_senior_rule
        and gemini_marked_high
    ):
        final_risk_tier = "High"

    # --------------------------------------------------------
    # 5. BILL VS INVOICE + CASH = DETERMINISTIC HIGH
    #
    # This rule can be independently validated using the
    # structured fields already available in POSException.
    # --------------------------------------------------------

    has_print_bill = (
        exception.number_of_bill_prints > 0
    )

    comparison_bill_amount: Optional[float] = None

    if (
        exception.rebilled_amount is not None
        and exception.number_of_bill_prints >= 2
    ):
        comparison_bill_amount = (
            exception.rebilled_amount
        )

    elif exception.initial_bill_amount is not None:
        comparison_bill_amount = (
            exception.initial_bill_amount
        )

    has_amount_mismatch = False

    if (
        comparison_bill_amount is not None
        and exception.final_invoice_amount is not None
    ):
        has_amount_mismatch = (
            abs(
                float(
                    exception.final_invoice_amount
                )
                - float(
                    comparison_bill_amount
                )
            )
            > 0.01
        )

    has_item_removal = bool(
        str(
            exception.items_removed or ""
        ).strip()
    )

    has_item_addition = bool(
        str(
            exception.items_added or ""
        ).strip()
    )

    has_discount_change = bool(
        str(
            exception.discount_changes or ""
        ).strip()
    )

    has_material_mismatch = any(
        (
            has_amount_mismatch,
            has_item_removal,
            has_item_addition,
            has_discount_change,
        )
    )

    final_tender = str(
        exception.final_tender or ""
    ).strip()

    has_cash_final_tender = bool(
        re.search(
            r"\bCASH\b",
            final_tender,
            re.IGNORECASE,
        )
    )

    has_bill_invoice_mismatch_rule = (
        "LAST PRINT BILL"
        in rule_triggered
        or "LAST BILL NOT TALLY INVOICE"
        in flag_categories
    )

    if (
        has_bill_invoice_mismatch_rule
        and has_print_bill
        and has_material_mismatch
        and has_cash_final_tender
    ):
        final_risk_tier = "High"

    # --------------------------------------------------------
    # 6. FINAL PRIORITY RESULT
    #
    # High > Medium > Low
    # --------------------------------------------------------

    exception.risk_tier = (
        final_risk_tier
    )

    return exception

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

        # ----------------------------------------------------
        # SEND EVERY VALID GROUPED SLIP TO GEMINI
        #
        # Python does not decide whether the Slip is normal,
        # suspicious, low-signal, or worth reviewing.
        #
        # Every transaction successfully grouped by the exact
        # explicit "Slip:" value is sent to Gemini.
        #
        # Gemini performs the actual exception screening using
        # the criteria defined in SYSTEM_PROMPT.
        # ----------------------------------------------------

        candidate_trails.append(
            (
                slip_number,
                trail,
            )
        )

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

    total_prompt_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

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
                batch_token_usage,
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
            total_prompt_tokens += (
                batch_token_usage[
                    "prompt_tokens"
                ]
            )

            total_output_tokens += (
                batch_token_usage[
                    "output_tokens"
                ]
            )

            total_tokens += (
                batch_token_usage[
                    "total_tokens"
                ]
            )

            logger.info(
                "Finished Gemini batch %s of %s in %.3f seconds",
                batch_number,
                len(batches),
                elapsed,
            )

            tier_adjusted_exceptions: List[
                POSException
            ] = []

            for exception in (
                batch_result.exceptions
            ):
                returned_slip = (
                    normalize_slip_number(
                        exception.slip_number
                    )
                )

                same_slip_blocks = (
                    grouped_slips.get(
                        returned_slip,
                        [],
                    )
                )

                invoice_data = (
                    extract_same_slip_invoice_data(
                        primary_slip_number=(
                            returned_slip
                        ),
                        blocks=same_slip_blocks,
                    )
                )

                # ------------------------------------------------
                # CORRECT ONLY INVOICE OUTPUT VALUES
                #
                # These values now come only from the INVOICE
                # document whose explicit Slip: equals the
                # returned/Primary Slip.
                # ------------------------------------------------

                exception.invoice_number = (
                    invoice_data[
                        "invoice_number"
                    ]
                )

                exception.final_invoice_amount = (
                    invoice_data[
                        "final_invoice_amount"
                    ]
                )

                exception.final_tender = (
                    invoice_data[
                        "final_tender"
                    ]
                )

                # Existing risk-tier function is unchanged.
                exception = (
                    enforce_risk_tier_rules(
                        exception
                    )
                )

                tier_adjusted_exceptions.append(
                    exception
                )
            all_exceptions.extend(
                tier_adjusted_exceptions
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
        "token_usage": {
            "prompt_tokens": (
                total_prompt_tokens
            ),
            "output_tokens": (
                total_output_tokens
            ),
            "total_tokens": (
                total_tokens
            ),
        },
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
            slip_number = (
                normalize_slip_number(
                    slip_number
                )
            )

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
# ASYNCHRONOUS JOB INFRASTRUCTURE
# ============================================================

_firestore_client = None
_storage_client = None
_tasks_client = None


def get_firestore_client():
    global _firestore_client

    if _firestore_client is None:
        _firestore_client = firestore.Client(
            project=GOOGLE_CLOUD_PROJECT
        )

    return _firestore_client


def get_storage_client():
    global _storage_client

    if _storage_client is None:
        _storage_client = storage.Client(
            project=GOOGLE_CLOUD_PROJECT
        )

    return _storage_client


def get_tasks_client():
    global _tasks_client

    if _tasks_client is None:
        _tasks_client = (
            tasks_v2.CloudTasksClient()
        )

    return _tasks_client


def get_job_document(
    job_id: str,
):
    return (
        get_firestore_client()
        .collection(JOB_COLLECTION)
        .document(job_id)
    )


def write_gzip_json_to_storage(
    object_name: str,
    payload: Dict,
) -> None:
    bucket = get_storage_client().bucket(
        JOB_BUCKET
    )

    blob = bucket.blob(
        object_name
    )

    json_bytes = json.dumps(
        payload,
        ensure_ascii=False,
    ).encode("utf-8")

    compressed_bytes = gzip.compress(
        json_bytes
    )

    blob.upload_from_string(
        compressed_bytes,
        content_type="application/gzip",
    )


def read_gzip_json_from_storage(
    object_name: str,
) -> Dict:
    bucket = get_storage_client().bucket(
        JOB_BUCKET
    )

    blob = bucket.blob(
        object_name
    )

    compressed_bytes = (
        blob.download_as_bytes()
    )

    json_bytes = gzip.decompress(
        compressed_bytes
    )

    return json.loads(
        json_bytes.decode("utf-8")
    )


def enqueue_analysis_task(
    job_id: str,
) -> str:
    tasks_client = get_tasks_client()

    queue_path = tasks_client.queue_path(
        GOOGLE_CLOUD_PROJECT,
        TASK_LOCATION,
        TASK_QUEUE,
    )

    worker_url = (
        f"{SERVICE_URL}"
        f"/internal/jobs/{job_id}/run"
    )

    task_body = json.dumps(
        {
            "job_id": job_id,
        }
    ).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
    }

    if WORKER_TOKEN:
        headers["X-Worker-Token"] = (
            WORKER_TOKEN
        )

    http_request = tasks_v2.HttpRequest(
        http_method=(
            tasks_v2.HttpMethod.POST
        ),
        url=worker_url,
        headers=headers,
        body=task_body,
    )

    if TASK_SERVICE_ACCOUNT:
        http_request.oidc_token = (
            tasks_v2.OidcToken(
                service_account_email=(
                    TASK_SERVICE_ACCOUNT
                ),
                audience=SERVICE_URL,
            )
        )

    task = tasks_v2.Task(
        http_request=http_request,
        dispatch_deadline=(
            duration_pb2.Duration(
                seconds=(
                    TASK_DISPATCH_DEADLINE_SECONDS
                )
            )
        ),
    )

    created_task = tasks_client.create_task(
        request={
            "parent": queue_path,
            "task": task,
        }
    )

    return created_task.name


# ============================================================
# START ASYNCHRONOUS ANALYSIS JOB
# POST /jobs
# ============================================================

@app.post("/jobs")
def start_analysis_job():
    job_id = ""

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
                "analyze_ej_batch",
            )
        ).strip()

        store = str(
            payload.get(
                "store",
                "",
            )
        ).strip()

        ej_files = payload.get(
            "ej_files",
            [],
        )

        approved_rules = payload.get(
            "approved_rules",
            [],
        )

        if mode != "analyze_ej_batch":
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "POST /jobs supports only "
                        "analyze_ej_batch."
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
                    "error": "ej_files is empty.",
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

        job_id = uuid.uuid4().hex

        input_object_name = (
            f"jobs/{job_id}/input.json.gz"
        )

        result_object_name = (
            f"jobs/{job_id}/result.json.gz"
        )

        job_payload = {
            "mode": mode,
            "store": store,
            "ej_files": ej_files,
            "approved_rules": approved_rules,
        }

        write_gzip_json_to_storage(
            input_object_name,
            job_payload,
        )

        job_document = get_job_document(
            job_id
        )

        job_document.set(
            {
                "job_id": job_id,
                "status": "queued",
                "store": store,
                "mode": mode,
                "input_object": (
                    input_object_name
                ),
                "result_object": (
                    result_object_name
                ),
                "files_received": len(
                    ej_files
                ),
                "exception_count": 0,
                "processing_seconds": None,
                "analysis_summary": None,
                "error": None,
                "task_name": None,
                "created_at": (
                    firestore.SERVER_TIMESTAMP
                ),
                "started_at": None,
                "completed_at": None,
            }
        )

        try:
            task_name = enqueue_analysis_task(
                job_id
            )

        except Exception:
            job_document.set(
                {
                    "status": "failed",
                    "error": (
                        "The background task could not "
                        "be queued. Check Cloud Tasks "
                        "configuration and permissions."
                    ),
                    "completed_at": (
                        firestore.SERVER_TIMESTAMP
                    ),
                },
                merge=True,
            )
            raise

        job_document.update(
            {
                "task_name": task_name,
            }
        )

        return jsonify(
            {
                "success": True,
                "job_id": job_id,
                "status": "queued",
                "files_received": len(
                    ej_files
                ),
            }
        ), 202

    except Exception as exc:
        logger.exception(
            "Failed to create EJ analysis job."
        )

        return jsonify(
            {
                "success": False,
                "job_id": job_id,
                "status": "failed",
                "error": str(exc),
            }
        ), 500


# ============================================================
# CLOUD TASKS WORKER
# POST /internal/jobs/<job_id>/run
# ============================================================

@app.post(
    "/internal/jobs/<job_id>/run"
)
def run_analysis_job(
    job_id: str,
):
    job_document = get_job_document(
        job_id
    )

    try:
        if WORKER_TOKEN:
            received_token = (
                request.headers.get(
                    "X-Worker-Token",
                    "",
                )
            )

            if received_token != WORKER_TOKEN:
                return jsonify(
                    {
                        "success": False,
                        "job_id": job_id,
                        "status": "failed",
                        "error": "Unauthorized worker.",
                    }
                ), 401

        job_snapshot = job_document.get()

        if not job_snapshot.exists:
            return jsonify(
                {
                    "success": False,
                    "job_id": job_id,
                    "status": "not_found",
                    "error": "Job was not found.",
                }
            ), 404

        job_data = job_snapshot.to_dict()

        current_status = str(
            job_data.get(
                "status",
                "",
            )
        ).strip().lower()

        # ----------------------------------------------------
        # DUPLICATE CLOUD TASK PROTECTION
        # ----------------------------------------------------

        if current_status == "completed":
            logger.info(
                "Job %s is already completed. "
                "Skipping duplicate execution.",
                job_id,
            )

            return jsonify(
                {
                    "success": True,
                    "job_id": job_id,
                    "status": "completed",
                    "message": (
                        "Job was already completed."
                    ),
                }
            ), 200

        if current_status == "processing":
            logger.info(
                "Job %s is already processing. "
                "Skipping duplicate execution.",
                job_id,
            )

            return jsonify(
                {
                    "success": True,
                    "job_id": job_id,
                    "status": "processing",
                    "message": (
                        "Job is already processing."
                    ),
                }
            ), 200

        if current_status == "failed":
            logger.info(
                "Job %s is already marked failed. "
                "Skipping duplicate execution.",
                job_id,
            )

            return jsonify(
                {
                    "success": False,
                    "job_id": job_id,
                    "status": "failed",
                    "retryable": False,
                    "error": job_data.get(
                        "error",
                        "Job previously failed.",
                    ),
                }
            ), 200

        job_document.update(
            {
                "status": "processing",
                "started_at": (
                    firestore.SERVER_TIMESTAMP
                ),
                "completed_at": None,
                "error": None,
            }
        )

        logger.info(
            "Job %s changed from %s to processing.",
            job_id,
            current_status,
        )
        
        job_payload = (
            read_gzip_json_from_storage(
                job_data["input_object"]
            )
        )

        store = str(
            job_payload.get(
                "store",
                "",
            )
        ).strip()

        ej_files = job_payload.get(
            "ej_files",
            [],
        )

        approved_rules = job_payload.get(
            "approved_rules",
            [],
        )

        worker_started = time.time()

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
            time.time() - worker_started,
            3,
        )

        result_payload = {
            "success": True,
            "mode": "analyze_ej_batch",
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

        write_gzip_json_to_storage(
            job_data["result_object"],
            result_payload,
        )

        job_document.update(
            {
                "status": "completed",
                "exception_count": len(
                    exceptions
                ),
                "processing_seconds": (
                    processing_seconds
                ),
                "analysis_summary": (
                    analysis_summary
                ),
                "completed_at": (
                    firestore.SERVER_TIMESTAMP
                ),
                "error": None,
            }
        )

        return jsonify(
            {
                "success": True,
                "job_id": job_id,
                "status": "completed",
                "exception_count": len(
                    exceptions
                ),
            }
        ), 200

    except Exception as exc:
        logger.exception(
            "EJ background analysis failed."
        )

        error_message = str(
            exc
        )

        normalized_error = (
            error_message.upper()
        )

        non_retryable_error = any(
            error_text in normalized_error
            for error_text in {
                "402",
                "PAYMENT REQUIRED",
                "PREPAYMENT CREDITS ARE DEPLETED",
                "PREPAID CREDITS ARE DEPLETED",
                "INVALID API KEY",
                "API KEY NOT VALID",
                "PERMISSION DENIED",
                "UNAUTHENTICATED",
            }
        )

        # ----------------------------------------------------
        # PERMANENT ERROR
        # Do not retry the Cloud Task
        # ----------------------------------------------------

        if non_retryable_error:
            job_document.set(
                {
                    "status": "failed",
                    "error": error_message,
                    "completed_at": (
                        firestore.SERVER_TIMESTAMP
                    ),
                },
                merge=True,
            )

            logger.error(
                "Job %s encountered a non-retryable "
                "Gemini error. Returning HTTP 200 "
                "to stop Cloud Tasks retries.",
                job_id,
            )

            return jsonify(
                {
                    "success": False,
                    "job_id": job_id,
                    "status": "failed",
                    "retryable": False,
                    "error": error_message,
                }
            ), 200

        # ----------------------------------------------------
        # TEMPORARY ERROR
        # Put job back into queued state so Cloud Tasks
        # can legitimately retry it
        # ----------------------------------------------------

        job_document.set(
            {
                "status": "queued",
                "error": error_message,
                "started_at": None,
                "completed_at": None,
            },
            merge=True,
        )

        logger.error(
            "Job %s encountered a potentially "
            "temporary error. Returning HTTP 500 "
            "so Cloud Tasks may retry.",
            job_id,
        )

        return jsonify(
            {
                "success": False,
                "job_id": job_id,
                "status": "queued",
                "retryable": True,
                "error": error_message,
            }
        ), 500


# ============================================================
# CHECK JOB STATUS
# GET /jobs/<job_id>
# ============================================================

@app.get("/jobs/<job_id>")
def check_analysis_job(
    job_id: str,
):
    try:
        job_snapshot = (
            get_job_document(
                job_id
            ).get()
        )

        if not job_snapshot.exists:
            return jsonify(
                {
                    "success": False,
                    "job_id": job_id,
                    "status": "not_found",
                    "error": "Job was not found.",
                }
            ), 404

        job_data = job_snapshot.to_dict()

        status = str(
            job_data.get(
                "status",
                "unknown",
            )
        )

        return jsonify(
            {
                "success": (
                    status != "failed"
                ),
                "job_id": job_id,
                "status": status,
                "files_received": (
                    job_data.get(
                        "files_received",
                        0,
                    )
                ),
                "exception_count": (
                    job_data.get(
                        "exception_count",
                        0,
                    )
                ),
                "processing_seconds": (
                    job_data.get(
                        "processing_seconds"
                    )
                ),
                "error": job_data.get(
                    "error"
                ),
            }
        ), 200

    except Exception as exc:
        logger.exception(
            "Failed to retrieve job status."
        )

        return jsonify(
            {
                "success": False,
                "job_id": job_id,
                "status": "failed",
                "error": str(exc),
            }
        ), 500


# ============================================================
# GET FINAL JOB RESULT
# GET /jobs/<job_id>/result
# ============================================================

@app.get(
    "/jobs/<job_id>/result"
)
def get_analysis_job_result(
    job_id: str,
):
    try:
        job_snapshot = (
            get_job_document(
                job_id
            ).get()
        )

        if not job_snapshot.exists:
            return jsonify(
                {
                    "success": False,
                    "job_id": job_id,
                    "status": "not_found",
                    "error": "Job was not found.",
                }
            ), 404

        job_data = job_snapshot.to_dict()

        status = str(
            job_data.get(
                "status",
                "unknown",
            )
        )

        if status == "failed":
            return jsonify(
                {
                    "success": False,
                    "job_id": job_id,
                    "status": "failed",
                    "error": job_data.get(
                        "error",
                        "Analysis failed.",
                    ),
                }
            ), 500

        if status != "completed":
            return jsonify(
                {
                    "success": True,
                    "job_id": job_id,
                    "status": status,
                    "message": (
                        "The analysis is not "
                        "complete yet."
                    ),
                }
            ), 202

        result_payload = (
            read_gzip_json_from_storage(
                job_data["result_object"]
            )
        )

        return jsonify(
            result_payload
        ), 200

    except Exception as exc:
        logger.exception(
            "Failed to retrieve job result."
        )

        return jsonify(
            {
                "success": False,
                "job_id": job_id,
                "status": "failed",
                "error": str(exc),
            }
        ), 500


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
            "job_endpoints": [
                "POST /jobs",
                "GET /jobs/<job_id>",
                "GET /jobs/<job_id>/result",
            ],
            "async_configuration": {
                "project": GOOGLE_CLOUD_PROJECT,
                "task_location": TASK_LOCATION,
                "task_queue": TASK_QUEUE,
                "job_bucket": JOB_BUCKET,
                "service_url": SERVICE_URL,
                "worker_token_configured": bool(
                    WORKER_TOKEN
                ),
                "task_service_account_configured": bool(
                    TASK_SERVICE_ACCOUNT
                ),
            },
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
            "job_endpoints": [
                "POST /jobs",
                "GET /jobs/<job_id>",
                "GET /jobs/<job_id>/result",
            ],
            "async_configuration": {
                "project": GOOGLE_CLOUD_PROJECT,
                "task_location": TASK_LOCATION,
                "task_queue": TASK_QUEUE,
                "job_bucket": JOB_BUCKET,
                "service_url": SERVICE_URL,
                "worker_token_configured": bool(
                    WORKER_TOKEN
                ),
                "task_service_account_configured": bool(
                    TASK_SERVICE_ACCOUNT
                ),
            },
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
