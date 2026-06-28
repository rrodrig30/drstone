"""Dr Stone point-of-care front-end: uric-acid stone probability from a
non-contrast CT (stone HU) + routine ED labs. Decision support only."""

from __future__ import annotations

import asyncio
import html
import json
import os
import sys

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from drstone import config as C
from drstone.predict import predict

router = APIRouter()

FRIENDLY = {
    "hu_peak": "Stone peak HU", "hu_mean": "Stone mean HU", "urine_ph": "Urine pH",
    "co2": "Bicarbonate (CO₂)", "cl": "Chloride", "anion_gap": "Anion gap",
    "bun": "BUN", "creatinine": "Creatinine", "ca": "Calcium", "glucose": "Glucose",
    "age": "Age", "gender_M": "Male sex",
}

PAGE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dr Stone — Uric-Acid Stone Probability</title>
<script src="/static/js/vendor/htmx.min.js"></script>
<style>
 body{margin:0;font-family:system-ui,sans-serif;background:#0e1217;color:#e8edf2}
 .wrap{max-width:880px;margin:0 auto;padding:24px}
 h1{margin:0 0 2px;font-size:24px} .sub{color:#9fb0c0;font-size:13px;margin-bottom:18px}
 .card{background:#161c24;border:1px solid #263445;border-radius:10px;padding:18px;margin-bottom:16px}
 h3{margin:0 0 10px;font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:#8fa6bd}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}
 label{display:block;font-size:12px;color:#aebccb;margin-bottom:3px}
 input,select{width:100%;box-sizing:border-box;background:#0e1217;color:#e8edf2;
  border:1px solid #2c3e52;border-radius:6px;padding:7px 8px;font-size:14px}
 .hint{font-size:11px;color:#6b7d8f;margin-top:8px}
 button{margin-top:14px;background:#2b6cb0;color:#fff;border:0;border-radius:7px;
  padding:10px 18px;font-size:15px;cursor:pointer} button:hover{background:#3182ce}
 .disclaimer{font-size:11px;color:#6b7d8f;margin-top:18px;line-height:1.5}
</style></head><body><div class="wrap">
<h1>Dr&nbsp;Stone</h1>
<div class="sub">Uric-acid stone probability from a non-contrast stone-protocol CT and routine labs — no dual-energy CT required. Decision support, not a substitute for stone analysis.</div>
<form hx-post="/api/drstone/predict" hx-target="#result" hx-swap="innerHTML">
 <div class="card"><h3>Imaging (from the NCCT)</h3>
   <div style="margin-bottom:12px">
     <label>Auto-measure from CT — DICOM series folder (on server, optional)</label>
     <div style="display:flex;gap:8px">
       <input name="dicom_path" type="text" placeholder="/path/to/dicom/series" style="flex:1">
       <button type="button" onclick="measureHU()" style="margin:0;background:#2f855a">Detect stones</button>
     </div>
     <div id="measure-status" class="hint"></div>
     <div id="stone-list"></div>
   </div>
   <div class="grid">
   <div><label>Stone peak HU</label><input name="hu_peak" type="number" step="1" placeholder="e.g. 640"></div>
   <div><label>Stone mean HU</label><input name="hu_mean" type="number" step="1" placeholder="e.g. 350"></div>
 </div></div>
 <div class="card"><h3>Labs &amp; demographics (leave unknown blank)</h3><div class="grid">
   <div><label>Urine pH</label><input name="urine_ph" type="number" step="0.1" placeholder="5.0-9.0"></div>
   <div><label>Sodium</label><input name="na" type="number" step="1" placeholder="mmol/L"></div>
   <div><label>Chloride</label><input name="cl" type="number" step="1" placeholder="mmol/L"></div>
   <div><label>Bicarbonate (CO₂)</label><input name="co2" type="number" step="1" placeholder="mmol/L"></div>
   <div><label>BUN</label><input name="bun" type="number" step="1"></div>
   <div><label>Creatinine</label><input name="creatinine" type="number" step="0.1"></div>
   <div><label>Calcium</label><input name="ca" type="number" step="0.1" placeholder="mg/dL"></div>
   <div><label>Glucose</label><input name="glucose" type="number" step="1"></div>
   <div><label>Age</label><input name="age" type="number" step="1"></div>
   <div><label>Sex</label><select name="sex"><option value="">—</option><option>Male</option><option>Female</option></select></div>
 </div>
 <div class="hint">Anion gap is computed from Na, Cl and CO₂. The model tolerates missing labs.</div>
 <button type="submit">Estimate UA probability</button></div>
</form>
<div id="result"></div>
<div class="disclaimer">For research/decision-support use. Single-energy CT cannot reliably separate calcium-oxalate from calcium-phosphate; this tool targets the actionable uric-acid vs non-uric-acid distinction. Confirm composition with stone analysis when available.</div>
</div>
<script>
function fmtVol(mm3){ return mm3>=1000 ? (mm3/1000).toFixed(1)+' cm³' : Math.round(mm3)+' mm³'; }
function pickStone(idx, stones){
  var s=stones[idx];
  document.getElementsByName('hu_peak')[0].value=Math.round(s.peak_hu);
  document.getElementsByName('hu_mean')[0].value=Math.round(s.mean_hu);
  var rows=document.querySelectorAll('.stone-opt');
  for(var i=0;i<rows.length;i++){ rows[i].style.borderColor = (i==idx)?'#2f855a':'#2c3e52'; rows[i].style.background=(i==idx)?'#16241b':'#0e1217'; }
  document.getElementById('measure-status').style.color='#2f855a';
  document.getElementById('measure-status').textContent='Selected '+s.location+' stone: peak '+Math.round(s.peak_hu)+' / mean '+Math.round(s.mean_hu)+' HU. Now add labs and estimate.';
}
window._stones=[];
function measureHU(){
  var path=document.getElementsByName('dicom_path')[0].value.trim();
  var st=document.getElementById('measure-status'); var list=document.getElementById('stone-list');
  list.innerHTML='';
  if(!path){st.style.color='#c05621';st.textContent='Enter a DICOM series folder path first.';return;}
  st.style.color='#9fb0c0'; st.textContent='Segmenting CT and detecting stones (may take ~30-60s)…';
  var fd=new FormData(); fd.append('dicom_path',path);
  fetch('/api/drstone/measure',{method:'POST',body:fd}).then(r=>r.json()).then(d=>{
    var stones=d.stones||[]; window._stones=stones;
    if(!stones.length){ st.style.color='#c05621'; st.textContent='No stone detected'+(d.error?(': '+d.error):'')+'. Enter HU manually if needed.'; return; }
    var html='';
    for(var i=0;i<stones.length;i++){ var s=stones[i];
      html += '<div class="stone-opt" onclick="pickStone('+i+',window._stones)" style="cursor:pointer;border:1px solid #2c3e52;border-radius:6px;padding:8px 10px;margin-top:6px;font-size:13px">'
        + '<b>'+(i+1)+'. '+s.location+'</b> — '+fmtVol(s.volume_mm3)+' — peak '+Math.round(s.peak_hu)+' / mean '+Math.round(s.mean_hu)+' HU</div>';
    }
    list.innerHTML=html;
    if(stones.length==1){ pickStone(0,stones); }
    else { st.style.color='#9fb0c0'; st.textContent='Detected '+stones.length+' stones — click the one you are evaluating.'; }
  }).catch(e=>{st.style.color='#c05621';st.textContent='Error: '+e;});
}
</script>
</body></html>"""


@router.get("/drstone", response_class=HTMLResponse)
def drstone_page():
    return HTMLResponse(PAGE)


@router.post("/api/drstone/measure")
async def drstone_measure(request: Request):
    """Auto-measure stone HU from a DICOM series, in an isolated subprocess."""
    form = dict(await request.form())
    path = str(form.get("dicom_path", "")).strip()
    if not path or not os.path.isdir(path):
        return JSONResponse({"found": False, "error": "path is not a directory on the server"},
                            status_code=400)
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "drstone.measure_cli", "--all", path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            cwd=C.REPO_ROOT)
        out, _ = await proc.communicate()
        line = out.decode().strip().splitlines()[-1] if out.strip() else "{}"
        return JSONResponse(json.loads(line))
    except Exception as e:
        return JSONResponse({"found": False, "error": str(e)}, status_code=500)


@router.post("/api/drstone/predict", response_class=HTMLResponse)
async def drstone_predict(request: Request):
    form = dict(await request.form())
    try:
        r = predict(form)
    except Exception as e:
        return HTMLResponse(f'<div class="card" style="border-color:#a33">Error: {html.escape(str(e))}</div>')
    p = r["probability"]; prev = r["prevalence"]
    pct = p * 100
    # Probabilities are compressed at this prevalence/AUC; interpret RELATIVE to
    # the population baseline and the high-sensitivity rule-out threshold. The
    # model's primary clinical value is ranking + ruling UA out (high NPV).
    if p < 0.02:
        band, color, msg = ("Uric acid effectively EXCLUDED", "#2f855a",
            "Very low probability — a low score reliably rules out uric acid (high negative predictive value). Proceed with standard (non-UA) management.")
    elif p < prev:
        band, color, msg = ("Uric acid LESS LIKELY than average", "#38a169",
            f"Below the population baseline ({prev*100:.0f}%). Uric acid is less likely than the average stone patient.")
    elif p < 2 * prev:
        band, color, msg = ("At / above average", "#b7791f",
            f"At or above the population baseline ({prev*100:.0f}%). Consider urine pH trend and clinical context; stone analysis if retrieved.")
    else:
        band, color, msg = ("Uric acid ELEVATED", "#c05621",
            "Well above baseline — consider a urine alkalinization trial and metabolic work-up; many uric-acid stones are medically dissolvable.")

    rows = ""
    for c in r["contributions"][:6]:
        if c["value"] != c["value"]:        # NaN -> not provided
            continue
        toward = c["shap"] > 0
        arrow = "▲ toward UA" if toward else "▼ away from UA"
        col = "#e06c75" if toward else "#61afef"
        val = c["value"]
        vals = f"{val:.1f}" if abs(val) < 1000 else f"{val:.0f}"
        rows += (f'<tr><td>{html.escape(FRIENDLY.get(c["feature"], c["feature"]))}</td>'
                 f'<td style="text-align:right">{vals}</td>'
                 f'<td style="color:{col}">{arrow}</td></tr>')

    auc_txt = f"{r['auc']:.2f}" if r['auc'] == r['auc'] else "—"
    return HTMLResponse(f"""
<div class="card">
  <h3>Estimated uric-acid probability</h3>
  <div style="font-size:34px;font-weight:700;color:{color}">{pct:.0f}%</div>
  <div style="background:#0e1217;border-radius:6px;height:14px;margin:8px 0;overflow:hidden;border:1px solid #2c3e52">
    <div style="width:{min(100,pct):.0f}%;height:100%;background:{color}"></div></div>
  <div style="font-weight:600;color:{color};margin:6px 0">{band}</div>
  <div style="font-size:13px;color:#c3d0dc;line-height:1.5">{msg}</div>
  <div style="font-size:12px;color:#8295a7;margin-top:8px">Population baseline ≈ {prev*100:.0f}% · model AUROC {auc_txt} (95% CI 0.68–0.87) · {r['n_provided']}/{r['n_features']} inputs provided</div>
  <h3 style="margin-top:16px">Why (per-case drivers)</h3>
  <table style="width:100%;border-collapse:collapse;font-size:13px">
    <thead><tr style="color:#8fa6bd"><th style="text-align:left">Feature</th><th style="text-align:right">Value</th><th style="text-align:left;padding-left:14px">Effect</th></tr></thead>
    <tbody>{rows or '<tr><td colspan=3 style="color:#8295a7">enter values above</td></tr>'}</tbody>
  </table>
</div>""")
