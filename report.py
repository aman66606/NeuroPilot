"""PDF research report (reportlab)."""
import io
from collections import Counter

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .core.signal import CLASSES

DISCLAIMER = ("NeuroPilot AI is an educational and research demonstration. Simulated neural signals are used by default. "
              "The application does not read private thoughts, diagnose neurological disorders, provide medical "
              "treatment or control actual medical devices.")


def _table(rows, widths=None):
    t = Table(rows, colWidths=widths)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    return t


def build_pdf(session, preds, feedback, metrics, retrain):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                            title=f"NeuroPilot AI report {session['id']}")
    st = getSampleStyleSheet()
    S = [Paragraph("NeuroPilot AI - Research Session Report", st["Title"]), Paragraph(DISCLAIMER, st["Italic"]), Spacer(1, 10),
         Paragraph("Session", st["Heading2"]),
         _table([["Field", "Value"], ["ID", session["id"]], ["Name", session["name"]], ["Status", session["status"]],
                 ["Created", session["created_at"]], ["Started", session["started_at"] or "-"], ["Ended", session["ended_at"] or "-"]], [4 * cm, 12 * cm])]
    n = len(preds)
    acc_n = sum(p["accepted"] for p in preds)
    tot = [p["total_ms"] for p in preds]
    fb_ok = sum(f["correct"] for f in feedback)
    S += [Spacer(1, 10), Paragraph("Closed-loop summary", st["Heading2"]),
          _table([["Metric", "Value"], ["Windows processed", n], ["Commands executed (above threshold)", acc_n],
                  ["Rejected (low confidence)", n - acc_n], ["Feedback events", len(feedback)],
                  ["Feedback agreement", f"{fb_ok}/{len(feedback)}" if feedback else "-"],
                  ["Mean total latency (ms)", f"{sum(tot) / n:.2f}" if n else "-"],
                  ["Max total latency (ms)", f"{max(tot):.2f}" if n else "-"]], [8 * cm, 8 * cm])]
    cnt = Counter(p["predicted"] for p in preds)
    d = Drawing(400, 160)
    ch = VerticalBarChart()
    ch.x, ch.y, ch.width, ch.height = 30, 20, 340, 120
    ch.data = [[cnt.get(c, 0) for c in CLASSES]]
    ch.categoryAxis.categoryNames = CLASSES
    ch.bars[0].fillColor = colors.HexColor("#06b6d4")
    d.add(ch)
    S += [Spacer(1, 10), Paragraph("Prediction distribution", st["Heading2"]), d]
    S += [Paragraph("Decoder evaluation (held-out simulated test set)", st["Heading2"]),
          Paragraph(f"Accuracy {metrics['accuracy'] * 100:.1f}% | macro F1 {metrics['macro_f1']:.3f} | "
                    f"ITR ~{metrics['itr_bits_min']:.0f} bits/min (theoretical, 1 s window). Trained on synthetic data only.", st["Normal"]),
          _table([["true \\ pred"] + CLASSES] + [[CLASSES[i]] + r for i, r in enumerate(metrics["confusion_matrix"])])]
    if retrain:
        S += [Spacer(1, 10), Paragraph("Adaptive retraining", st["Heading2"]),
              Paragraph(f"Field test at {retrain['field_noise']} uV noise: {retrain['before']['accuracy'] * 100:.1f}% before, "
                        f"{retrain['after']['accuracy'] * 100:.1f}% after using {retrain['verified_used']} verified examples.", st["Normal"])]
    doc.build(S)
    return buf.getvalue()
