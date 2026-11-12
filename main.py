import json
import logging
import os
import time
from typing import List, Optional

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

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-3.8-flash",
).strip()

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not configured.")


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

    flag_categories: List[str] = Field(default_factory=list)

    risk_tier: str = ""
    finding: str = ""
    evidence_trail: str = ""
    rule_triggered: str = ""


class EJAnalysis(BaseModel):
    exceptions: List[POSException] = Field(default_factory=list)


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_gemini_client():
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is missing."
        )

    return genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a POS Electronic Journal transaction exception analyst.

Your task is to analyze restaurant POS Electronic Journal (EJ)
printer-trail data and identify transaction trails that may require
human investigation.

A flagged transaction is a POTENTIAL IRREGULARITY ONLY.

Never state or imply that fraud, theft, misconduct, or financial loss
definitely occurred.

============================================================
PRIMARY TRANSACTION IDENTIFIER
============================================================

Slip Number is the primary transaction-tracing identifier.

For every unique Slip Number:

1. Locate every event containing that exact Slip Number.
2. Reconstruct its transaction activity chronologically.
3. Analyze the complete available trail before deciding whether an
   exception exists.
4. Do not flag an isolated event without considering the surrounding
   transaction activity.

============================================================
DOCUMENT CLASSIFICATION
============================================================

BAR, WINDOW, NACHO, KITCHEN, MEAT, MIDDLE and similar preparation or
production-printer records are NOT customer Print Bills.

Repeated production-printer records for the same Slip Number DO NOT
constitute Multiple Print Bill.

Customer-facing records such as:

- BILL
- PRINT BILL
- REBILL
- REPRINT
- INVOICE
- BILL CANCELLATION
- RETURN
- REFUND

must be identified from actual EJ document headings and corroborating
fields.

Never classify a production-print record as a customer bill merely
because it has a Slip Number.

============================================================
TRANSACTION INFORMATION
============================================================

When available, determine:

- Slip Number
- Transaction Number
- Bill Slip Number
- Invoice Number
- Return Slip Number
- POS ID / Terminal
- Staff
- Table
- Sales Type
- Business Date
- timestamps
- customer bill amounts
- item changes
- discounts
- tender/payment activity
- cancellations
- returns/refunds
- invoice
- final settlement

============================================================
BILL AMOUNT DEFINITIONS
============================================================

Initial Bill Amount:
The amount on the FIRST identifiable CUSTOMER Print Bill.

Rebilled Amount:
The amount on the LAST identifiable CUSTOMER Print Bill before final
invoice or settlement.

Final Invoice Amount:
The amount finally invoiced or settled.

Amount Difference:
Rebilled Amount minus Initial Bill Amount.

Number of Bill Prints:
The number of CUSTOMER Print Bills only.

Never count BAR, WINDOW, NACHO, KITCHEN, MEAT, MIDDLE or similar
production-printer records as customer Print Bills.

Bill Amount Trail:
Preserve every identifiable customer Print Bill amount in chronological
order.

============================================================
ITEM CHANGES
============================================================

Identify items removed or added AFTER an earlier CUSTOMER Print Bill
when the EJ evidence supports that conclusion.

Do not classify ordinary progressive kitchen ordering before the first
customer Print Bill as suspicious item removal/addition.

============================================================
DISCOUNTS
============================================================

Identify:

- Initial discount
- Final discount
- Discounts added after Print Bill
- Discounts removed after Print Bill
- Discounts replaced or changed
- Multiple discount combinations

The existence of a discount by itself is NOT an exception.

============================================================
TENDER
============================================================

Trace tender/payment activity when available.

Tender changes or multiple tender are NOT automatically irregular.

Flag them only where the complete sequence warrants human review.

============================================================
POTENTIAL EXCEPTION PATTERNS
============================================================

Consider patterns including:

1. Multiple customer Print Bills where the amount materially changes.

2. Customer Print Bill followed by item removal.

3. Customer Print Bill followed by material item addition or
   substitution requiring review.

4. Bill cancellation followed by replacement/rebill.

5. Repeated bill cancellations.

6. Material amount decrease between the first customer Print Bill and
   last customer Print Bill.

7. Discount added, removed, or materially changed after an earlier
   customer Print Bill.

8. Multiple or unusual discount combinations requiring review.

9. VAT-exempt treatment combined with another discount where the
   transaction sequence warrants review.

10. Tender/payment changes or reversals associated with other unusual
    activity.

11. Return/refund with an unusual preceding or subsequent trail.

12. Material transaction changes after apparent settlement.

13. Reprints associated with other material changes.

14. Customer Print Bill remaining outstanding for an unusually long
    period where the supplied evidence supports this conclusion.

15. Other materially inconsistent transaction sequences that warrant
    human investigation.

============================================================
DO NOT FLAG SOLELY BECAUSE
============================================================

Do NOT flag a transaction solely because:

- the same Slip Number occurs many times;
- production printer records repeat;
- BAR prints repeat;
- WINDOW prints repeat;
- NACHO prints repeat;
- KITCHEN prints repeat;
- MEAT prints repeat;
- a transaction crosses midnight;
- a discount exists;
- multiple tender exists;
- the same unchanged customer Bill was reprinted.

============================================================
EVIDENCE STANDARD
============================================================

Every exception must be grounded in specific evidence from the supplied
EJ.

The evidence_trail must explain the relevant sequence chronologically.

Do not invent:

- Slip Numbers
- transactions
- amounts
- timestamps
- items
- documents
- cancellations
- discounts
- tenders
- invoices
- returns
- refunds

If a value cannot be identified, return an empty string or null as
appropriate.

============================================================
RISK TIER
============================================================

Use exactly one:

High
Medium
Low

HIGH:
Strong multi-event exception sequence or high-impact control concern.

MEDIUM:
Meaningful transaction anomaly requiring investigation but without
sufficient evidence to classify High.

LOW:
Weak but identifiable anomaly worth monitoring.

Do not assign a Risk Tier to transactions that are not exceptions.

============================================================
OUTPUT
============================================================

Return ONLY exceptions that warrant human review.

If no meaningful exception exists, return an empty exceptions array.
"""


# ============================================================
# ANALYSIS
# ============================================================

def analyze_ej(
    store: str,
    ej_file_name: str,
    ej_content: str,
    approved_rules=None,
):
    if approved_rules is None:
        approved_rules = []

    client = get_gemini_client()

    user_prompt = f"""
STORE:
{store}

SOURCE EJ FILE:
{ej_file_name}

APPROVED LEGITIMATE-PATTERN RULES:
{json.dumps(approved_rules, ensure_ascii=False)}

IMPORTANT:
Approved rules may suppress a matching legitimate pattern only when
the complete rule conditions are satisfied.

An approved rule must NOT suppress a separate material anomaly in the
same transaction.

EJ CONTENT START
============================================================
{ej_content}
============================================================
EJ CONTENT END

Analyze this EJ and return the structured exception result.
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

    elapsed = round(time.time() - started, 3)

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    try:
        result = EJAnalysis.model_validate_json(response.text)
    except Exception as exc:
        logger.exception("Gemini response failed validation.")
        raise RuntimeError(
            f"Invalid structured response from Gemini: {exc}"
        ) from exc

    return result, elapsed


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
    try:
        payload = request.get_json(silent=True)

        if not isinstance(payload, dict):
            return jsonify(
                {
                    "success": False,
                    "error": "Request body must be valid JSON.",
                }
            ), 400

        mode = str(
            payload.get("mode", "analyze_ej")
        ).strip()

        if mode != "analyze_ej":
            return jsonify(
                {
                    "success": False,
                    "error": f"Unsupported mode: {mode}",
                }
            ), 400

        store = str(
            payload.get("store", "")
        ).strip()

        ej_file_name = str(
            payload.get("ej_file_name", "")
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
                    "error": "ej_file_name is required.",
                }
            ), 400

        if not isinstance(ej_content, str):
            return jsonify(
                {
                    "success": False,
                    "error": "ej_content must be text.",
                }
            ), 400

        if not ej_content.strip():
            return jsonify(
                {
                    "success": False,
                    "error": "ej_content is empty.",
                }
            ), 400

        if not isinstance(approved_rules, list):
            return jsonify(
                {
                    "success": False,
                    "error": "approved_rules must be an array.",
                }
            ), 400

        # ----------------------------------------------------
        # GEMINI ANALYSIS
        # ----------------------------------------------------

        analysis, elapsed = analyze_ej(
            store=store,
            ej_file_name=ej_file_name,
            ej_content=ej_content,
            approved_rules=approved_rules,
        )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify(
            {
                "success": True,
                "mode": "analyze_ej",
                "model": GEMINI_MODEL,
                "store": store,
                "ej_file_name": ej_file_name,
                "exception_count": len(
                    analysis.exceptions
                ),
                "exceptions": [
                    item.model_dump()
                    for item in analysis.exceptions
                ],
                "processing_seconds": elapsed,
            }
        ), 200

    except Exception as exc:
        logger.exception("EJ analysis failed.")

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
