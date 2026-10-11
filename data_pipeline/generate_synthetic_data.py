"""Generate synthetic enterprise tickets, paired diagnostic images and a knowledge base.

Outputs (under config.DATA_DIR):
    tickets.jsonl                 3,000+ tickets with the required schema
    kb/knowledge_base.jsonl       knowledge-base articles referenced by ``knowledge_base_ids``
    images/<VISUAL_CLASS>/*.png   synthetic attachments for CNN training

Ticket schema:
    {id, ticket_text, image_path, customer_tier, intent, priority,
     target_department, knowledge_base_ids}
plus two helper fields used downstream: ``visual_label`` (CNN ground truth, null when
there is no attachment) and ``difficulty`` ("easy" | "hard").

Usage:
    python -m data_pipeline.generate_synthetic_data --n 3200 --seed 42
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import config

# --------------------------------------------------------------------------- #
# Knowledge-base catalogue: one article per (intent, sub-topic)
# --------------------------------------------------------------------------- #
CATALOG: dict[str, list[dict[str, Any]]] = {
    "ui_bug": [
        {"title": "Dashboard widgets render blank after a release",
         "symptoms": ["the dashboard widgets render as blank white panels",
                      "charts spin forever and never load data",
                      "the analytics page shows no data tiles"],
         "steps": ["Hard refresh the browser and clear cached static assets.",
                   "Confirm the frontend build hash matches the latest release on the status page.",
                   "Capture the browser console output and the failing network request.",
                   "Disable the dashboard_v3 feature flag for the affected tenant."]},
        {"title": "Form submit button unresponsive or rejecting valid input",
         "symptoms": ["the submit button does nothing when clicked",
                      "the form freezes after pressing save",
                      "the checkout form throws a validation error on valid input"],
         "steps": ["Reproduce in a private window with extensions disabled.",
                   "Check the console for uncaught validation exceptions.",
                   "Verify the locale and date format settings for the user profile.",
                   "File a frontend defect with the HAR file attached."]},
        {"title": "Layout broken on mobile and small screens",
         "symptoms": ["the layout overlaps and buttons are cut off on mobile",
                      "text is truncated and the menu is unreachable on small screens",
                      "the sidebar covers the main content on tablets"],
         "steps": ["Record the device model, OS version and browser version.",
                   "Test with the responsive viewport emulator at the reported width.",
                   "Check whether the compact navigation flag is enabled.",
                   "Attach screenshots and route to Frontend Engineering."]},
        {"title": "File upload dialog fails with a front-end error",
         "symptoms": ["file upload shows a red error banner instantly",
                      "the upload dialog closes without saving the file",
                      "drag and drop upload raises a request failed popup"],
         "steps": ["Confirm the file type and size are within the documented limits.",
                   "Retry using the browse button instead of drag and drop.",
                   "Inspect the failing upload request status code in the network tab.",
                   "Escalate with the request id from the error banner."]},
    ],
    "hardware_failure": [
        {"title": "Device boot loop or kernel panic after startup",
         "symptoms": ["the device keeps rebooting and shows a kernel panic",
                      "the controller is stuck in a boot loop after the firmware update",
                      "the unit crashes on startup with a memory dump on screen"],
         "steps": ["Power cycle the unit for thirty seconds with all peripherals unplugged.",
                   "Boot into recovery mode and reflash the last known good firmware.",
                   "Collect the serial console log and the crash dump photo.",
                   "Open an RMA if the panic repeats after a clean reflash."]},
        {"title": "Overheating and thermal shutdown",
         "symptoms": ["the unit overheats and shuts itself down",
                      "the fan runs at full speed and the device powers off",
                      "thermal warning LEDs are flashing red before shutdown"],
         "steps": ["Verify ambient temperature and airflow clearance around the unit.",
                   "Clean the intake filter and confirm the fan spins freely.",
                   "Read the thermal log and note the peak temperature reading.",
                   "Replace the fan module or open an RMA when peaks exceed 85 C."]},
        {"title": "Sensor or peripheral module not detected",
         "symptoms": ["the sensor module is not detected by the hub",
                      "the USB peripheral disconnects randomly",
                      "the camera module reports no signal"],
         "steps": ["Reseat the module and inspect the connector for bent pins.",
                   "Swap the cable and test the module on a second port.",
                   "Update the hub firmware to the latest stable version.",
                   "Open an RMA when the module fails on a known good hub."]},
        {"title": "Power supply and battery faults",
         "symptoms": ["the battery drains within minutes",
                      "the power supply makes a buzzing noise and the device will not charge",
                      "the unit does not power on at all"],
         "steps": ["Test with a known good power adapter and outlet.",
                   "Check the battery health report and cycle count.",
                   "Stop using the unit immediately if the casing is swollen or hot.",
                   "Ship a replacement power supply and open an RMA for the battery."]},
    ],
    "billing_issue": [
        {"title": "Duplicate charge on an invoice",
         "symptoms": ["we were charged twice for the same invoice",
                      "a duplicate payment appears on our statement",
                      "the receipt shows the subscription billed two times"],
         "steps": ["Compare both transaction ids in the payment ledger.",
                   "Confirm that one charge is a pending authorization that will drop off.",
                   "Issue a refund for the confirmed duplicate settlement.",
                   "Send the customer a corrected receipt."]},
        {"title": "Incorrect tax or amount on an invoice",
         "symptoms": ["the tax amount on the invoice looks wrong",
                      "the invoice total does not match our contract price",
                      "we were billed for seats we already removed"],
         "steps": ["Check the billing country and tax id on the account.",
                   "Compare invoice line items with the active contract.",
                   "Recalculate proration for mid-cycle seat changes.",
                   "Issue a credit note and regenerate the invoice."]},
        {"title": "Refund approved but not received",
         "symptoms": ["the refund we were promised has not arrived",
                      "our refund request was approved but no money came back",
                      "the credit note is missing from the account"],
         "steps": ["Verify the refund status and the processor reference id.",
                   "Explain that card refunds take five to ten business days.",
                   "Re-issue the refund through bank transfer if the card is closed.",
                   "Escalate to Billing & Finance if the processor reports a failure."]},
        {"title": "Payment method declined at renewal",
         "symptoms": ["our card keeps getting declined at renewal",
                      "the payment fails with an unknown error",
                      "the subscription was suspended after a failed payment"],
         "steps": ["Ask the customer to update the payment method in the billing portal.",
                   "Retry the charge after confirming the new method is verified.",
                   "Extend the grace period by seven days for enterprise accounts.",
                   "Reactivate the subscription once payment succeeds."]},
    ],
    "account_access": [
        {"title": "Password reset email not arriving",
         "symptoms": ["the password reset email never arrives",
                      "the reset link says it has expired immediately",
                      "we cannot receive the verification code"],
         "steps": ["Check the spam folder and the corporate mail gateway quarantine.",
                   "Confirm the email address on file and resend the reset link.",
                   "Ask the customer to use the link within fifteen minutes.",
                   "Allow-list the sender domain with the customer IT team."]},
        {"title": "Locked out by multi-factor authentication",
         "symptoms": ["we are locked out after the MFA prompt",
                      "the authenticator code is always rejected",
                      "the MFA device was lost and we cannot sign in"],
         "steps": ["Verify identity through the account owner and the registered phone.",
                   "Check device clock drift, which breaks time based codes.",
                   "Issue a one time recovery code to the verified owner.",
                   "Require MFA re-enrolment at the next sign in."]},
        {"title": "Single sign-on redirect loop or assertion error",
         "symptoms": ["single sign-on redirects back to the login page in a loop",
                      "SAML login fails with an invalid assertion error",
                      "SSO works for some users but not others"],
         "steps": ["Compare the identity provider certificate with the one uploaded.",
                   "Check the NameID format and the audience restriction value.",
                   "Test with the SAML tracer and capture the assertion.",
                   "Fall back to a temporary password login for the admin."]},
        {"title": "Account locked or suspended",
         "symptoms": ["the account is locked after several login attempts",
                      "the admin account shows as suspended",
                      "new team members cannot be activated"],
         "steps": ["Review the lock reason in the admin audit log.",
                   "Unlock the account after verifying the owner.",
                   "Check the seat limit and licence status for new users.",
                   "Advise a password change if the lock came from repeated failures."]},
    ],
    "performance_degradation": [
        {"title": "API latency spikes",
         "symptoms": ["API responses take over ten seconds",
                      "latency has tripled since yesterday",
                      "requests time out intermittently"],
         "steps": ["Check the regional status page for active incidents.",
                   "Compare p95 latency against the trailing seven day baseline.",
                   "Look for a noisy neighbour or a recent traffic burst on the tenant.",
                   "Page the on-call SRE if p95 stays above the SLO for ten minutes."]},
        {"title": "Reports and queries running slowly",
         "symptoms": ["reports take minutes to generate",
                      "the search query runs extremely slowly",
                      "the export job never finishes"],
         "steps": ["Identify the slow query id and its filter set.",
                   "Narrow the date range and retry to confirm data volume is the cause.",
                   "Check whether the tenant index rebuild is running.",
                   "Open a performance ticket with the query plan attached."]},
        {"title": "Elevated 5xx error rate",
         "symptoms": ["we are seeing a wave of 502 errors",
                      "the service returns 503 under load",
                      "error rates jumped above five percent"],
         "steps": ["Confirm the error rate on the service dashboard.",
                   "Check recent deploys and roll back if the spike lines up.",
                   "Scale the affected deployment and watch the autoscaler.",
                   "Open a major incident if the rate exceeds ten percent."]},
        {"title": "Sync and queue backlog",
         "symptoms": ["data sync is hours behind",
                      "the job queue is backed up and nothing is processing",
                      "webhooks are delivered with a long delay"],
         "steps": ["Check queue depth and consumer lag on the worker dashboard.",
                   "Restart stuck consumers and clear poison messages.",
                   "Increase worker replicas temporarily.",
                   "Replay missed webhooks once the backlog drains."]},
    ],
    "security_incident": [
        {"title": "Suspicious login activity",
         "symptoms": ["there are logins from countries we do not operate in",
                      "we saw unrecognized sign-ins on an admin account",
                      "multiple failed logins came from an unknown IP"],
         "steps": ["Force sign out all sessions for the affected users.",
                   "Reset credentials and require MFA re-enrolment.",
                   "Export the sign-in audit log for the last thirty days.",
                   "Block the offending IP ranges and notify Security Operations."]},
        {"title": "Leaked API key or secret",
         "symptoms": ["an API key was accidentally committed to a public repo",
                      "we think a secret token has been exposed",
                      "our key shows usage we did not make"],
         "steps": ["Revoke the exposed key immediately.",
                   "Issue a replacement key with the minimum required scopes.",
                   "Review usage logs since the time of exposure.",
                   "Ask the customer to purge the key from git history."]},
        {"title": "Phishing report involving our brand",
         "symptoms": ["staff received a phishing email that impersonates your company",
                      "a fake login page that looks like yours is circulating",
                      "users reported a suspicious payment request email"],
         "steps": ["Collect the full email headers and the phishing URL.",
                   "Submit the URL to the takedown and blocklist service.",
                   "Warn affected customers with the approved advisory.",
                   "Track the report in the security incident register."]},
        {"title": "Possible data exposure",
         "symptoms": ["a shared link exposes internal documents",
                      "we found customer data visible to the wrong tenant",
                      "an export file was sent to an unauthorized recipient"],
         "steps": ["Disable the shared link or export immediately.",
                   "Preserve audit logs and do not delete any evidence.",
                   "Determine scope: which records and which recipients.",
                   "Engage the incident commander and the privacy officer."]},
    ],
    "feature_request": [
        {"title": "Dark mode and custom theming requests",
         "symptoms": ["please add a dark mode option",
                      "it would help to have custom theme colors",
                      "can you support branded themes for our portal"],
         "steps": ["Thank the customer and log the request on the product board.",
                   "Link the request to the theming epic and add the account.",
                   "Share the public roadmap link when the item is planned.",
                   "Do not promise delivery dates."]},
        {"title": "Bulk export and API additions",
         "symptoms": ["we need a bulk export for all records",
                      "please add an endpoint to list audit events",
                      "it would be great to have CSV export in reports"],
         "steps": ["Confirm whether an existing endpoint or export already covers the need.",
                   "Share the API changelog and the export documentation.",
                   "Log the gap with a clear use case on the product board.",
                   "Offer a workaround with the existing pagination API."]},
        {"title": "Granular roles and permissions requests",
         "symptoms": ["we need finer grained roles for contractors",
                      "please allow read only access per project",
                      "can admins restrict who can delete items"],
         "steps": ["Explain the current role model and its limits.",
                   "Suggest a custom role workaround if available on the plan.",
                   "Log the request under the access control epic.",
                   "Flag enterprise accounts to the product manager."]},
        {"title": "New integrations requests",
         "symptoms": ["please add a Slack integration",
                      "we would love a Jira two way sync",
                      "can you support Microsoft Teams notifications"],
         "steps": ["Check the integrations catalogue for an existing connector.",
                   "Point to webhooks as an interim option.",
                   "Log the request and count the interested accounts.",
                   "Share the partner programme link for custom integrations."]},
    ],
    "spam_or_irrelevant": [
        {"title": "Promotional spam handling",
         "symptoms": ["CONGRATULATIONS you have been selected to win a free prize",
                      "buy cheap followers and boost your ranking today",
                      "limited offer 90 percent off click the link now"],
         "steps": ["Do not reply or click any links.",
                   "Mark the sender as spam and close the ticket.",
                   "Add the sender domain to the blocklist.",
                   "Report repeated abuse to Trust & Safety."]},
        {"title": "Phishing link bait in tickets",
         "symptoms": ["verify your wallet now to claim your reward",
                      "your parcel is waiting click here to confirm payment",
                      "urgent your mailbox is full login to avoid deletion"],
         "steps": ["Never open attachments or links from the ticket.",
                   "Quarantine the ticket and its attachment.",
                   "Forward the sample to Security Operations for analysis.",
                   "Block the submitting account."]},
        {"title": "Gibberish and test submissions",
         "symptoms": ["asdf qwer zxcv test test",
                      "hello hello is anyone there lorem ipsum",
                      "just checking if this form works"],
         "steps": ["Close the ticket with the test submission macro.",
                   "Do not count it towards SLA reporting.",
                   "Invite the user to resubmit with details.",
                   "Escalate only if the volume suggests automation."]},
        {"title": "Off-topic solicitations",
         "symptoms": ["we offer SEO services for your website",
                      "looking for investors for a crypto project",
                      "do you want to buy a mailing list"],
         "steps": ["Close the ticket as irrelevant.",
                   "Do not share any internal contact details.",
                   "Add the sender to the solicitation filter.",
                   "Report persistent senders to Trust & Safety."]},
    ],
}

DIAGNOSIS = {
    "ui_bug": "Start by reproducing the issue in a clean browser profile and note the build version.",
    "hardware_failure": "Collect the device model, firmware version and any on-screen fault code before acting.",
    "billing_issue": "Verify the invoice id, the payment processor reference and the active contract.",
    "account_access": "Verify the identity of the requester before changing any credential or lock state.",
    "performance_degradation": "Compare current latency and error metrics with the trailing baseline for the tenant.",
    "security_incident": "Treat the report as real until disproven and preserve all audit evidence.",
    "feature_request": "Capture the business use case and the number of affected users.",
    "spam_or_irrelevant": "Confirm the ticket contains no actionable support request and no safe attachment.",
}
ESCALATION = {
    "ui_bug": "the defect reproduces on the latest build for more than one user",
    "hardware_failure": "the fault persists after a clean firmware reflash",
    "billing_issue": "the amount in dispute exceeds the standard refund limit",
    "account_access": "identity cannot be verified or the account shows signs of takeover",
    "performance_degradation": "the p95 latency or error rate breaches the service level objective",
    "security_incident": "any credential, key or customer data may be exposed",
    "feature_request": "an enterprise customer links the request to a renewal",
    "spam_or_irrelevant": "the sender repeats the abuse or targets other customers",
}

# --------------------------------------------------------------------------- #
# Ticket text building blocks
# --------------------------------------------------------------------------- #
PRODUCTS = ["Nimbus Console", "Orbit Dashboard", "Atlas Portal", "Helix Mobile App", "Vertex Gateway"]
DEVICES = ["NX-400 edge gateway", "VX-220 sensor hub", "RT-9 rack controller", "LP-15 field tablet"]
BILLING_TARGETS = ["our Pro subscription", "the annual plan", "the invoice portal", "our enterprise contract"]
WHEN = ["this morning", "since yesterday", "after the last update", "about two hours ago",
        "since Monday", "over the weekend", "after we changed settings"]
OPENERS = ["Hi team,", "Hello support,", "Good morning,", "Hi,", "Dear support team,", "Hey,"]
SIGN_OFFS = ["Thanks, Priya", "Regards, Marcus", "Thank you, Aiko", "Cheers, Daniel",
             "Best, Sofia", "Thanks, Ravi", "Regards, Elena"]
ASKS = ["Please advise.", "Can someone look into this?", "What are the next steps?",
        "Let me know what you need from us.", "Any update would be appreciated."]
SYMPTOM_FRAMES = [
    "We noticed that {s} on {t}.",
    "Writing because {s} on {t} {w}.",
    "Problem report: {s} ({t}, {w}).",
    "{S} on {t}. It started {w}.",
]
CONTEXT_LINES = [
    "Reference code {code}.",
    "Our account id is {acct}.",
    "Roughly {n} users are affected.",
    "We already tried restarting and it did not help.",
    "This is our second report about it.",
]
DISTRACTOR_FRAMES = ["Separately, I also think {s}.", "Not sure if related, but {s}.", "By the way, {s}."]
URGENCY_PHRASES = {
    0: ["Production is completely down and customers are impacted.",
        "This is a total outage for our whole team.",
        "Critical: all operations are stopped right now."],
    1: ["This is blocking our release this week.",
        "It is seriously slowing down our team today.",
        "We need a fix soon because it affects a key workflow."],
    2: ["It is not blocking us, but it is quite annoying.",
        "There is a workaround, but we would like a proper fix.",
        "Moderate impact for now."],
    3: ["Low priority, just wanted to flag it.",
        "No rush on this one.",
        "Not urgent at all, whenever you have time."],
}
SPAM_TAILS = ["Reply now.", "Visit http://promo-{n}.example/claim today.", "No obligation, act fast!",
              "Offer ends tonight."]
ATTACH_PHRASES = {
    "UI_ERROR": ["I attached a screenshot of the error.", "Please see the attached screen capture."],
    "HARDWARE_FAULT": ["I attached a photo of the crash dump on the display.", "See the attached picture of the unit."],
    "VALID_DOCUMENT": ["I attached the receipt and a log extract.", "The invoice PDF is attached.",
                       "Attached is the document with the details."],
    "SPAM_IMAGE": ["See the image below.", "Check our banner."],
}

INTENT_WEIGHTS = {
    "ui_bug": 0.16, "hardware_failure": 0.13, "billing_issue": 0.15, "account_access": 0.15,
    "performance_degradation": 0.12, "security_incident": 0.09, "feature_request": 0.12,
    "spam_or_irrelevant": 0.08,
}
BASE_SEVERITY = {  # 0 = P1 ... 3 = P4
    "ui_bug": 2, "hardware_failure": 1, "billing_issue": 2, "account_access": 2,
    "performance_degradation": 2, "security_incident": 1, "feature_request": 3,
    "spam_or_irrelevant": 3,
}
URGENCY_SHIFT = {0: -2, 1: -1, 2: 0, 3: 1}
TIER_BONUS = {"free": 0, "pro": 0, "enterprise": 1}

INTENT_VISUAL = {
    "ui_bug": [("UI_ERROR", 0.90), ("VALID_DOCUMENT", 0.10)],
    "hardware_failure": [("HARDWARE_FAULT", 0.92), ("VALID_DOCUMENT", 0.08)],
    "billing_issue": [("VALID_DOCUMENT", 0.95), ("UI_ERROR", 0.05)],
    "account_access": [("UI_ERROR", 0.80), ("VALID_DOCUMENT", 0.20)],
    "performance_degradation": [("VALID_DOCUMENT", 0.60), ("UI_ERROR", 0.25), ("HARDWARE_FAULT", 0.15)],
    "security_incident": [("VALID_DOCUMENT", 0.70), ("UI_ERROR", 0.30)],
    "feature_request": [("VALID_DOCUMENT", 0.60), ("UI_ERROR", 0.40)],
    "spam_or_irrelevant": [("SPAM_IMAGE", 1.0)],
}
IMAGE_PROBABILITY = {"feature_request": 0.50, "spam_or_irrelevant": 0.85}
DEFAULT_IMAGE_PROBABILITY = 0.92


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


# --------------------------------------------------------------------------- #
# Knowledge base
# --------------------------------------------------------------------------- #
def build_knowledge_base() -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Return (articles, intent -> ordered list of kb ids)."""
    articles: list[dict[str, Any]] = []
    by_intent: dict[str, list[str]] = {intent: [] for intent in config.INTENTS}
    counter = 1
    for intent in config.INTENTS:
        dept = config.INTENT_TO_DEPARTMENT[intent]
        for sub in CATALOG[intent]:
            kb_id = f"KB-{counter:04d}"
            counter += 1
            steps = "\n".join(f"{i}. {step}" for i, step in enumerate(sub["steps"], start=1))
            content = (
                f"Overview: {sub['title']}. This article covers {intent.replace('_', ' ')} cases "
                f"owned by {dept}.\n\n"
                f"Typical symptoms reported by customers: {'; '.join(sub['symptoms'])}.\n\n"
                f"Diagnosis: {DIAGNOSIS[intent]}\n\n"
                f"Resolution steps:\n{steps}\n\n"
                f"Escalation: route to {dept} when {ESCALATION[intent]}. "
                f"Enterprise tier customers have a one hour response target for critical tickets, "
                f"pro tier four hours and free tier twenty four hours."
            )
            articles.append({
                "kb_id": kb_id, "title": sub["title"], "intent": intent,
                "department": dept, "content": content,
                "tags": [intent, *sub["title"].lower().split()[:3]],
            })
            by_intent[intent].append(kb_id)
    return articles, by_intent


# --------------------------------------------------------------------------- #
# Image rendering
# --------------------------------------------------------------------------- #
def _big_text(img: Image.Image, xy: tuple[int, int], text: str, scale: int, fill: tuple[int, int, int]) -> None:
    font = ImageFont.load_default()
    left, top, right, bottom = ImageDraw.Draw(img).textbbox((0, 0), text, font=font)
    layer = Image.new("L", (right - left + 2, bottom - top + 2), 0)
    ImageDraw.Draw(layer).text((-left + 1, -top + 1), text, fill=255, font=font)
    layer = layer.resize((layer.width * scale, layer.height * scale), Image.Resampling.NEAREST)
    img.paste(Image.new("RGB", layer.size, fill), xy, layer)


def _draw_ui_error(rng: random.Random, s: int) -> Image.Image:
    bg = rng.choice([(245, 245, 248), (32, 34, 40), (236, 240, 247)])
    img = Image.new("RGB", (s, s), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, s, 18], fill=rng.choice([(52, 86, 160), (70, 70, 80), (30, 120, 100)]))
    for i in range(3):
        d.ellipse([6 + i * 12, 6, 12 + i * 12, 12], fill=(255, 255, 255))
    for _ in range(rng.randint(3, 6)):
        x0, y0 = rng.randint(6, s // 2), rng.randint(26, s - 60)
        d.rectangle([x0, y0, x0 + rng.randint(30, 90), y0 + rng.randint(8, 24)], outline=(150, 150, 165))
    w, h = rng.randint(130, 165), rng.randint(72, 92)
    x0 = (s - w) // 2 + rng.randint(-8, 8)
    y0 = (s - h) // 2 + rng.randint(-8, 8)
    accent = rng.choice([(200, 40, 40), (220, 120, 30)])
    d.rectangle([x0, y0, x0 + w, y0 + h], fill=(255, 255, 255), outline=accent, width=2)
    d.rectangle([x0, y0, x0 + w, y0 + 14], fill=accent)
    d.text((x0 + 4, y0 + 2), "Error", fill=(255, 255, 255))
    d.ellipse([x0 + 8, y0 + 24, x0 + 28, y0 + 44], fill=accent)
    d.text((x0 + 16, y0 + 29), "!", fill=(255, 255, 255))
    d.text((x0 + 36, y0 + 24), f"ERR_{rng.randint(1000, 9999)}", fill=(30, 30, 30))
    d.text((x0 + 36, y0 + 38), rng.choice(["Request failed", "Page crashed", "Load failed",
                                           "Null reference", "Render timeout", "Bad response"]),
           fill=(70, 70, 70))
    d.rectangle([x0 + w - 52, y0 + h - 20, x0 + w - 8, y0 + h - 6], fill=(60, 110, 200))
    d.text((x0 + w - 42, y0 + h - 18), "Retry", fill=(255, 255, 255))
    return img


def _hw_terminal(rng: random.Random, s: int) -> Image.Image:
    img = Image.new("RGB", (s, s), (8, 10, 12))
    d = ImageDraw.Draw(img)
    y = 4
    header = rng.choice(["KERNEL PANIC: fatal exception", f"FATAL: machine check 0x{rng.randint(0, 255):02x}",
                         "BUG: unable to handle fault", f"SEGFAULT at 0x{rng.randint(0, 65535):04x}"])
    d.text((4, y), header, fill=(255, 70, 70))
    y += 13
    d.text((4, y), "Call Trace:", fill=(230, 230, 230))
    y += 12
    for _ in range(rng.randint(3, 5)):
        d.text((8, y), f"[<{rng.getrandbits(32):08x}>] fn_{rng.randint(1, 99)}+0x{rng.randint(16, 4095):x}",
               fill=(200, 200, 200))
        y += 11
    y += 4
    d.text((4, y), "Memory dump:", fill=(230, 230, 230))
    y += 12
    while y < s - 12:
        row = " ".join(f"{rng.randint(0, 255):02x}" for _ in range(6))
        d.text((4, y), f"{rng.getrandbits(32):08x}: {row}", fill=(80, 220, 120))
        y += 11
    return img


def _hw_pcb(rng: random.Random, s: int) -> Image.Image:
    img = Image.new("RGB", (s, s), (18, 82, 48))
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(25, 40)):
        x, y = rng.randint(0, s), rng.randint(0, s)
        pts = [(x, y)]
        for _ in range(rng.randint(2, 4)):
            if rng.random() < 0.5:
                x = max(0, min(s, x + rng.randint(-60, 60)))
            else:
                y = max(0, min(s, y + rng.randint(-60, 60)))
            pts.append((x, y))
        d.line(pts, fill=(205, 175, 60), width=2)
    for _ in range(rng.randint(2, 4)):
        x0, y0 = rng.randint(10, s - 70), rng.randint(10, s - 70)
        w, h = rng.randint(30, 60), rng.randint(30, 60)
        d.rectangle([x0, y0, x0 + w, y0 + h], fill=(28, 28, 32))
        for k in range(0, w, 6):
            d.line([x0 + k, y0 - 3, x0 + k, y0], fill=(200, 200, 200))
            d.line([x0 + k, y0 + h, x0 + k, y0 + h + 3], fill=(200, 200, 200))
    cx, cy = rng.randint(40, s - 40), rng.randint(40, s - 40)
    for radius, color in zip((30, 22, 14, 7), ((255, 220, 80), (255, 120, 30), (160, 20, 10), (30, 8, 8))):
        d.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)
    d.text((6, s - 14), "OVERTEMP", fill=(255, 255, 255))
    return img


def _draw_hardware(rng: random.Random, s: int) -> Image.Image:
    return _hw_terminal(rng, s) if rng.random() < 0.55 else _hw_pcb(rng, s)


def _draw_spam(rng: random.Random, s: int) -> Image.Image:
    def bright() -> tuple[int, int, int]:
        return rng.randint(120, 255), rng.randint(0, 255), rng.randint(0, 255)

    c1, c2 = np.array(bright(), dtype=np.float32), np.array(bright(), dtype=np.float32)
    t = np.linspace(0, 1, s, dtype=np.float32)[None, :, None]
    arr = c1[None, None, :] * (1 - t) + c2[None, None, :] * t
    img = Image.fromarray(np.repeat(arr, s, axis=0).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(6, 12)):
        x0, y0 = rng.randint(0, s - 20), rng.randint(0, s - 20)
        box = [x0, y0, x0 + rng.randint(15, 80), y0 + rng.randint(15, 80)]
        if rng.random() < 0.5:
            d.ellipse(box, fill=bright())
        else:
            d.rectangle(box, fill=bright())
    cx, cy = rng.randint(60, s - 60), rng.randint(50, s - 90)
    pts = []
    for i in range(24):
        radius = 55 if i % 2 == 0 else 32
        angle = i * np.pi / 12
        pts.append((cx + radius * float(np.cos(angle)), cy + radius * float(np.sin(angle))))
    d.polygon(pts, fill=(255, 230, 0), outline=(220, 0, 0))
    text = rng.choice(["WIN $$$", "FREE!!", "CLICK", "90% OFF", "PRIZE", "BONUS"])
    _big_text(img, (max(4, (s - len(text) * 24) // 2), s - 70), text, 4, (255, 255, 255))
    cell = 5
    for gx in range(8):
        for gy in range(8):
            if rng.random() < 0.5:
                d.rectangle([6 + gx * cell, 6 + gy * cell, 6 + (gx + 1) * cell, 6 + (gy + 1) * cell],
                            fill=(0, 0, 0))
    return img


def _doc_receipt(rng: random.Random, s: int) -> Image.Image:
    g = rng.randint(190, 232)
    img = Image.new("RGB", (s, s), (g, g, g))
    d = ImageDraw.Draw(img)
    x0, width = (s - 140) // 2, 140
    d.rectangle([x0, 6, x0 + width, s - 6], fill=(255, 255, 250))
    d.text((x0 + 10, 12), f"ACME STORE #{rng.randint(1, 99)}", fill=(20, 20, 20))
    y, total = 30, 0.0
    for _ in range(rng.randint(5, 9)):
        price = rng.uniform(2, 60)
        total += price
        d.text((x0 + 8, y), rng.choice(["Subscription", "Seat", "Support", "Storage", "Add-on", "Credit"]),
               fill=(40, 40, 40))
        d.text((x0 + 100, y), f"{price:6.2f}", fill=(40, 40, 40))
        y += 11
    d.line([x0 + 8, y + 2, x0 + width - 8, y + 2], fill=(60, 60, 60))
    d.text((x0 + 8, y + 6), "TOTAL", fill=(0, 0, 0))
    d.text((x0 + 100, y + 6), f"{total:6.2f}", fill=(0, 0, 0))
    bx = x0 + 10
    while bx < x0 + width - 10:
        bw = rng.randint(1, 3)
        d.rectangle([bx, s - 40, bx + bw, s - 14], fill=(0, 0, 0))
        bx += bw + rng.randint(1, 3)
    return img


def _doc_invoice(rng: random.Random, s: int) -> Image.Image:
    img = Image.new("RGB", (s, s), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, s, 34], fill=rng.choice([(40, 70, 140), (50, 110, 90), (90, 60, 130)]))
    _big_text(img, (8, 8), "INVOICE", 2, (255, 255, 255))
    d.text((8, 42), f"No. INV-{rng.randint(100000, 999999)}", fill=(40, 40, 40))
    top = 62
    for row in range(6):
        y = top + row * 20
        d.line([8, y, s - 8, y], fill=(150, 150, 150))
        if row < 5:
            d.text((12, y + 5), rng.choice(["Platform fee", "Extra seats", "Premium support", "Overage", "Tax"]),
                   fill=(50, 50, 50))
            d.text((s - 60, y + 5), f"{rng.uniform(10, 900):7.2f}", fill=(50, 50, 50))
    for x in (8, 120, s - 8):
        d.line([x, top, x, top + 100], fill=(150, 150, 150))
    d.rectangle([s - 90, s - 40, s - 10, s - 14], outline=(40, 40, 40))
    d.text((s - 84, s - 33), "PAID", fill=(30, 130, 60))
    return img


def _doc_log(rng: random.Random, s: int) -> Image.Image:
    dark = rng.random() < 0.4
    bg, fg = ((25, 28, 35), (170, 205, 170)) if dark else ((250, 250, 250), (60, 60, 70))
    img = Image.new("RGB", (s, s), bg)
    d = ImageDraw.Draw(img)
    y = 4
    while y < s - 12:
        level = rng.choices(["INFO", "DEBUG", "WARN"], weights=[0.7, 0.2, 0.1])[0]
        stamp = f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"
        svc = rng.choice(["api", "auth", "sync", "db", "cache"])
        color = (200, 160, 40) if level == "WARN" else fg
        d.text((4, y), f"{stamp} {level:<5} {svc} request ok {rng.randint(10, 400)}ms", fill=color)
        y += 11
    return img


def _draw_document(rng: random.Random, s: int) -> Image.Image:
    return rng.choice([_doc_receipt, _doc_invoice, _doc_log])(rng, s)


RENDERERS = {
    "UI_ERROR": _draw_ui_error,
    "HARDWARE_FAULT": _draw_hardware,
    "SPAM_IMAGE": _draw_spam,
    "VALID_DOCUMENT": _draw_document,
}


def render_image(label: str, rng: random.Random, size: int) -> Image.Image:
    img = RENDERERS[label](rng, size)
    angle = rng.uniform(-2.5, 2.5)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=img.getpixel((0, 0)))
    if rng.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    nrng = np.random.default_rng(rng.getrandbits(32))
    arr = np.asarray(img, dtype=np.float32) + nrng.normal(0, rng.uniform(2, 10), (size, size, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# --------------------------------------------------------------------------- #
# Tickets
# --------------------------------------------------------------------------- #
def compose_text(rng: random.Random, intent: str, sub: dict[str, Any], urgency: int,
                 hard: bool, visual: str | None) -> str:
    symptom = rng.choice(sub["symptoms"])
    if intent == "spam_or_irrelevant":
        parts = [_cap(symptom) + ".", rng.choice(SPAM_TAILS).format(n=rng.randint(100, 9999))]
        if visual:
            parts.append(rng.choice(ATTACH_PHRASES[visual]))
        return " ".join(parts)

    if hard:
        target = "the system"
    elif intent == "hardware_failure":
        target = rng.choice(DEVICES)
    elif intent == "billing_issue":
        target = rng.choice(BILLING_TARGETS)
    else:
        target = rng.choice(PRODUCTS)
    when = rng.choice(WHEN)
    parts: list[str] = []
    if rng.random() < 0.7:
        parts.append(rng.choice(OPENERS))
    parts.append(rng.choice(SYMPTOM_FRAMES).format(s=symptom, S=_cap(symptom), t=target, w=when))
    if hard:
        other_intent = rng.choice([i for i in config.INTENTS if i not in (intent, "spam_or_irrelevant")])
        other = rng.choice(CATALOG[other_intent])
        parts.append(rng.choice(DISTRACTOR_FRAMES).format(s=rng.choice(other["symptoms"])))
    else:
        slots = {"code": f"{rng.choice(['ERR', 'E', 'SRV'])}-{rng.randint(1000, 9999)}",
                 "acct": f"ACC-{rng.randint(10000, 99999)}", "n": rng.randint(2, 400)}
        for line in rng.sample(CONTEXT_LINES, k=rng.randint(1, 2)):
            parts.append(line.format(**slots))
    if visual:
        parts.append(rng.choice(ATTACH_PHRASES[visual]))
    parts.append(rng.choice(URGENCY_PHRASES[urgency]))
    parts.append(rng.choice(ASKS))
    if rng.random() < 0.5:
        parts.append(rng.choice(SIGN_OFFS))
    return " ".join(parts)


def make_ticket(index: int, rng: random.Random, kb_by_intent: dict[str, list[str]],
                image_size: int, write_images: bool) -> dict[str, Any]:
    ticket_id = f"TKT-{index:06d}"
    intent = rng.choices(config.INTENTS, weights=[INTENT_WEIGHTS[i] for i in config.INTENTS])[0]
    sub_idx = rng.randrange(len(CATALOG[intent]))
    sub = CATALOG[intent][sub_idx]
    tier = rng.choices(config.CUSTOMER_TIERS, weights=config.TIER_WEIGHTS)[0]
    hard = intent != "spam_or_irrelevant" and rng.random() < config.HARD_TICKET_RATE

    if intent in ("feature_request", "spam_or_irrelevant"):
        urgency = rng.choice([2, 3])
    else:
        urgency = rng.choices([0, 1, 2, 3], weights=[0.12, 0.30, 0.33, 0.25])[0]
    if intent == "spam_or_irrelevant":
        sev = 3
    else:
        sev = BASE_SEVERITY[intent] + URGENCY_SHIFT[urgency] - TIER_BONUS[tier]
        sev = max(0, min(3, sev))

    visual: str | None = None
    if rng.random() < IMAGE_PROBABILITY.get(intent, DEFAULT_IMAGE_PROBABILITY):
        labels, probs = zip(*INTENT_VISUAL[intent])
        visual = rng.choices(labels, weights=probs)[0]

    image_path = None
    if visual:
        image_path = f"images/{visual}/{ticket_id}.png"
        img_seed = rng.getrandbits(48)
        if write_images:
            out = config.DATA_DIR / image_path
            out.parent.mkdir(parents=True, exist_ok=True)
            render_image(visual, random.Random(img_seed), image_size).save(out, format="PNG")

    primary = kb_by_intent[intent][sub_idx]
    kb_ids = [primary]
    if rng.random() < 0.35:
        related = [k for k in kb_by_intent[intent] if k != primary]
        kb_ids.append(rng.choice(related))

    return {
        "id": ticket_id,
        "ticket_text": compose_text(rng, intent, sub, urgency, hard, visual),
        "image_path": image_path,
        "customer_tier": tier,
        "intent": intent,
        "priority": config.PRIORITIES[sev],
        "target_department": config.INTENT_TO_DEPARTMENT[intent],
        "knowledge_base_ids": kb_ids,
        "visual_label": visual,
        "difficulty": "hard" if hard else "easy",
    }


def generate(n: int, seed: int, image_size: int, write_images: bool) -> list[dict[str, Any]]:
    config.ensure_dirs()
    rng = random.Random(seed)
    articles, kb_by_intent = build_knowledge_base()
    config.write_jsonl(config.KB_PATH, articles)
    tickets = [make_ticket(i, rng, kb_by_intent, image_size, write_images) for i in range(1, n + 1)]
    config.write_jsonl(config.TICKETS_PATH, tickets)
    return tickets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=config.N_TICKETS, help="number of tickets (>= 3000 recommended)")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--no-images", action="store_true", help="skip rendering PNG attachments")
    args = parser.parse_args()

    tickets = generate(args.n, args.seed, args.image_size, write_images=not args.no_images)
    print(f"tickets written : {len(tickets)} -> {config.TICKETS_PATH}")
    print(f"kb articles     : {len(build_knowledge_base()[0])} -> {config.KB_PATH}")
    for field in ("intent", "priority", "customer_tier", "visual_label", "difficulty"):
        counts = Counter(str(t[field]) for t in tickets)
        print(f"{field:14s}: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
