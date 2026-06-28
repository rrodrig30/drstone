"""Dr Stone point-of-care front-end: stone-composition probability distribution
+ acute management (MET vs. intervention) + tailored prevention, from a
non-contrast CT (stone HU/size/location) + routine ED labs. Decision support only."""

from __future__ import annotations

import asyncio
import html
import json
import os
import sys

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from drstone import config as C
from drstone import auth
from drstone.pages import topbar
from drstone.predict import predict, compose_assess

router = APIRouter()


def _guard_json(request: Request):
    """Return (user, None) if authenticated, else (None, 401 JSONResponse)."""
    user = auth.current_user(request)
    if user is None:
        return None, JSONResponse({"found": False, "error": "Not authenticated"},
                                  status_code=401)
    return user, None

FRIENDLY = {
    "hu_peak": "Stone peak HU", "hu_mean": "Stone mean HU", "urine_ph": "Urine pH",
    "co2": "Bicarbonate (CO₂)", "cl": "Chloride", "anion_gap": "Anion gap",
    "bun": "BUN", "creatinine": "Creatinine", "ca": "Calcium", "glucose": "Glucose",
    "age": "Age", "gender_M": "Male sex",
}

PAGE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dr Stone — Stone Composition &amp; Management</title>
<link rel="icon" href="/static/img/logo.png">
<link rel="stylesheet" href="/static/css/drstone.css">
<script src="/static/js/vendor/htmx.min.js"></script>
<style>
 .stone-opt{cursor:pointer;border:1px solid var(--line-2);border-radius:8px;
   padding:8px 10px;margin-top:6px;font-size:13px;background:#fff}
 .stone-opt:hover{border-color:var(--blue)}
 button[type=button]{margin-top:0}
</style></head><body>
<!--TOPBAR-->
<div class="wrap">
<h1>New assessment</h1>
<div class="sub">Stone composition probabilities + acute management (pass vs. treat) + tailored prevention, from a non-contrast stone-protocol CT and routine ED labs — no dual-energy CT required. Decision support / patient education, not a substitute for stone analysis or urology consultation.</div>
<form hx-post="/api/drstone/predict" hx-target="#result" hx-swap="innerHTML">
 <div class="card"><h3>Patient lookup (research)</h3>
   <div style="display:flex;gap:8px">
     <input name="mrn" type="text" placeholder="UT MRN" style="flex:1">
     <button type="button" onclick="loadLabs()" style="margin:0">Load labs</button>
   </div>
   <div id="lookup-status" class="hint"></div>
   <div class="hint">Auto-fills labs + the CT path from the project dataset (production would pull these from the EHR via FHIR/HL7).</div>
 </div>
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
   <div><label>Stone size (max mm)</label><input name="stone_size_mm" type="number" step="0.1" placeholder="e.g. 7"></div>
   <div><label>Location</label><select name="location"><option value="">—</option><option>Renal</option><option>Ureteral</option><option>Bladder</option></select></div>
 </div>
 <input type="hidden" name="hu_p95"><input type="hidden" name="volume_mm3">
 </div>
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
 <label style="display:flex;align-items:center;gap:8px;margin-top:12px;font-size:13px;color:#b03030;cursor:pointer">
   <input type="checkbox" name="infection" value="on" style="width:auto;margin:0">
   Suspected infection / obstruction (fever, pyuria, ↑WBC) — flag urologic emergency
 </label>
 <button type="submit">Assess stone &amp; recommend</button></div>
</form>
<div id="result"></div>
<div class="disclaimer">For research/decision-support use; not a substitute for stone analysis, a 24-hour urine metabolic evaluation, or urology consultation. Single-energy CT cannot reliably separate calcium-oxalate from calcium-phosphate — read the composition output as a ranked probability distribution and confirm with stone analysis when available. Acute and prevention guidance is draft content pending clinician sign-off.</div>
</div>
<script>
function fmtVol(mm3){ return mm3>=1000 ? (mm3/1000).toFixed(1)+' cm³' : Math.round(mm3)+' mm³'; }
function setVal(name,v){ var el=document.getElementsByName(name)[0]; if(el) el.value=v; }
function mapLoc(loc){ loc=(loc||'').toLowerCase();
  if(loc.indexOf('bladder')>=0) return 'Bladder';
  if(loc.indexOf('kidney')>=0) return 'Renal';
  return 'Ureteral'; }
function pickStone(idx, stones){
  var s=stones[idx];
  setVal('hu_peak',Math.round(s.peak_hu)); setVal('hu_mean',Math.round(s.mean_hu));
  setVal('hu_p95',Math.round(s.p95_hu||s.peak_hu)); setVal('volume_mm3',Math.round(s.volume_mm3));
  setVal('stone_size_mm',(s.max_diameter_mm||0).toFixed(1)); setVal('location',mapLoc(s.location));
  var rows=document.querySelectorAll('.stone-opt');
  for(var i=0;i<rows.length;i++){ rows[i].style.borderColor = (i==idx)?'#2f855a':'#cfd9e4'; rows[i].style.background=(i==idx)?'#eafaf0':'#fff'; }
  document.getElementById('measure-status').style.color='#2f855a';
  document.getElementById('measure-status').textContent='Selected '+s.location+' stone: peak '+Math.round(s.peak_hu)+' / mean '+Math.round(s.mean_hu)+' HU. Now add labs and estimate.';
}
window._stones=[];
function measureHU(){
  var path=document.getElementsByName('dicom_path')[0].value.trim();
  var st=document.getElementById('measure-status'); var list=document.getElementById('stone-list');
  list.innerHTML='';
  if(!path){st.style.color='#c05621';st.textContent='Enter a DICOM series folder path first.';return;}
  st.style.color='#5b6b7c'; st.textContent='Segmenting CT and detecting stones (may take ~30-60s)…';
  var fd=new FormData(); fd.append('dicom_path',path);
  fetch('/api/drstone/measure',{method:'POST',body:fd}).then(r=>r.json()).then(d=>{
    var stones=d.stones||[]; window._stones=stones;
    if(!stones.length){ st.style.color='#c05621'; st.textContent='No stone detected'+(d.error?(': '+d.error):'')+'. Enter HU manually if needed.'; return; }
    var html='';
    for(var i=0;i<stones.length;i++){ var s=stones[i];
      html += '<div class="stone-opt" onclick="pickStone('+i+',window._stones)">'
        + '<b>'+(i+1)+'. '+s.location+'</b> — '+fmtVol(s.volume_mm3)+' — peak '+Math.round(s.peak_hu)+' / mean '+Math.round(s.mean_hu)+' HU</div>';
    }
    list.innerHTML=html;
    if(stones.length==1){ pickStone(0,stones); }
    else { st.style.color='#5b6b7c'; st.textContent='Detected '+stones.length+' stones — click the one you are evaluating.'; }
  }).catch(e=>{st.style.color='#c05621';st.textContent='Error: '+e;});
}
function loadLabs(){
  var mrn=document.getElementsByName('mrn')[0].value.trim();
  var st=document.getElementById('lookup-status');
  if(!mrn){st.style.color='#c05621';st.textContent='Enter a UT MRN.';return;}
  st.style.color='#5b6b7c';st.textContent='Looking up…';
  var fd=new FormData();fd.append('mrn',mrn);
  fetch('/api/drstone/labs',{method:'POST',body:fd}).then(r=>r.json()).then(d=>{
    if(!d.found){st.style.color='#c05621';st.textContent='MRN not found in dataset.';return;}
    var L=d.labs;
    for(var k in L){ var el=document.getElementsByName(k)[0];
      if(el && L[k]!==null && L[k]!==undefined && L[k]!==''){ el.value=L[k]; } }
    if(d.dicom_path){ document.getElementsByName('dicom_path')[0].value=d.dicom_path; }
    st.style.color='#2f855a';
    st.textContent='Labs loaded'+(d.dicom_path?' + CT path filled — click Detect stones.':'. Add stone HU.');
  }).catch(e=>{st.style.color='#c05621';st.textContent='Error: '+e;});
}
</script>
</body></html>"""


@router.get("/drstone", response_class=HTMLResponse)
def drstone_page(request: Request):
    user = auth.current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    return HTMLResponse(PAGE.replace("<!--TOPBAR-->", topbar(user)))


@router.post("/api/drstone/labs")
async def drstone_labs(request: Request):
    """Auto-fill routine labs + CT path for a patient (research lookup)."""
    _user, err = _guard_json(request)
    if err:
        return err
    form = dict(await request.form())
    mrn = str(form.get("mrn", "")).strip()
    if not mrn:
        return JSONResponse({"found": False, "error": "no MRN"}, status_code=400)
    from drstone.lookup import lookup
    try:
        return JSONResponse(lookup(mrn))
    except Exception as e:
        return JSONResponse({"found": False, "error": str(e)}, status_code=500)


@router.post("/api/drstone/measure")
async def drstone_measure(request: Request):
    """Auto-measure stone HU from a DICOM series, in an isolated subprocess."""
    _user, err = _guard_json(request)
    if err:
        return err
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


TYPE_META = {
    "CaOx":     ("Calcium oxalate",      "#2c7fb8"),
    "CaP":      ("Calcium phosphate",    "#2e93a6"),
    "UA":       ("Uric acid",            "#c2741f"),
    "Struvite": ("Struvite (infection)", "#8059b8"),
    "Other":    ("Other / uncommon",     "#66788a"),
}
TIER_COLOR = {"met": "#2f855a", "surveillance": "#2b6cb0",
              "intervention": "#c05621", "info": "#66788a"}


def _li(items):
    return "".join(f"<li>{html.escape(x)}</li>" for x in items)


@router.post("/api/drstone/predict", response_class=HTMLResponse)
async def drstone_predict(request: Request):
    if auth.current_user(request) is None:
        return HTMLResponse('<div class="card" style="border-color:#c53030;color:#a02020">'
                            'Your session has expired. <a href="/login">Sign in again</a>.</div>',
                            status_code=401)
    form = dict(await request.form())
    try:
        r = compose_assess(form)
    except Exception as e:
        return HTMLResponse(f'<div class="card" style="border-color:#a33">Error: {html.escape(str(e))}</div>')

    # ---- composition distribution bars ---------------------------------
    bars = ""
    for d in r["distribution"]:
        label, col = TYPE_META.get(d["type"], (d["type"], "#66788a"))
        pct = d["p"] * 100
        bars += (
            f'<div style="margin:8px 0">'
            f'<div style="display:flex;justify-content:space-between;font-size:13px;color:#3a4858">'
            f'<span>{html.escape(label)}</span><span style="font-weight:700;color:{col}">{pct:.0f}%</span></div>'
            f'<div style="background:#eef2f6;border-radius:5px;height:12px;overflow:hidden;border:1px solid #dde4ec;margin-top:3px">'
            f'<div style="width:{min(100,pct):.0f}%;height:100%;background:{col}"></div></div></div>')
    top_labels = ", ".join(TYPE_META.get(t, (t, ""))[0] for t in r["top"])

    # ---- acute panel ---------------------------------------------------
    ac = r["acute"]
    acol = TIER_COLOR.get(ac["tier"], "#66788a")
    redflags = ""
    if ac["redflags"]:
        redflags = ('<div style="background:#fdecec;border:1px solid #f3c0c0;border-radius:8px;'
                    'padding:10px 12px;margin:10px 0">'
                    '<div style="color:#c53030;font-weight:700;font-size:13px">⚠ Red flags</div>'
                    f'<ul style="margin:5px 0 0;padding-left:18px;color:#8f2424;font-size:13px;line-height:1.55">{_li(ac["redflags"])}</ul></div>')
    acute_details = (f'<ul style="margin:8px 0 0;padding-left:18px;color:#3a4858;font-size:13.5px;line-height:1.6">{_li(ac["details"])}</ul>'
                     if ac["details"] else "")

    # ---- prevention panel ----------------------------------------------
    prev = r["prevention"]
    blocks = ""
    for b in prev["blocks"]:
        _, col = TYPE_META.get(b["type"], (b["label"], "#66788a"))
        blocks += (
            f'<div style="border:1px solid #e3e9f0;border-left:4px solid {col};border-radius:8px;padding:10px 13px;margin:10px 0;background:#fbfcfe">'
            f'<div style="font-weight:700;color:{col};margin-bottom:4px">{html.escape(b["label"])}</div>'
            f'<div style="font-size:11px;color:#8aa0b5;margin-top:8px;text-transform:uppercase;letter-spacing:.05em;font-weight:700">Diet</div>'
            f'<ul style="margin:3px 0 0;padding-left:18px;font-size:13px;color:#3a4858;line-height:1.55">{_li(b["diet"])}</ul>'
            f'<div style="font-size:11px;color:#8aa0b5;margin-top:8px;text-transform:uppercase;letter-spacing:.05em;font-weight:700">Medication</div>'
            f'<ul style="margin:3px 0 0;padding-left:18px;font-size:13px;color:#3a4858;line-height:1.55">{_li(b["meds"])}</ul>'
            f'<div style="font-size:11px;color:#8aa0b5;margin-top:8px;text-transform:uppercase;letter-spacing:.05em;font-weight:700">Lifestyle</div>'
            f'<ul style="margin:3px 0 0;padding-left:18px;font-size:13px;color:#3a4858;line-height:1.55">{_li(b["lifestyle"])}</ul>'
            f'</div>')
    flags = ""
    if prev["flags"]:
        flags = ('<div style="background:#eaf3fb;border:1px solid #cfe0f4;border-radius:8px;padding:10px 12px;margin:10px 0">'
                 '<div style="color:#2b6cb0;font-weight:700;font-size:13px">Metabolic flags (spot labs)</div>'
                 f'<ul style="margin:5px 0 0;padding-left:18px;color:#34506a;font-size:13px;line-height:1.55">{_li(prev["flags"])}</ul></div>')

    return HTMLResponse(f"""
<div class="card" style="border:1px dashed #d9a441;background:#fff8e9;box-shadow:none">
  <div style="font-size:12.5px;color:#946200;line-height:1.5">⚠ {html.escape(r["draft"])}</div>
</div>

<div class="card">
  <h3>Likely stone composition</h3>
  <div style="font-size:12.5px;color:#788798;margin-bottom:8px">Probability distribution from CT stone density + routine labs ({r['n_provided']} inputs). Single-energy CT cannot fully separate calcium subtypes — read as a ranked distribution, confirm with stone analysis.</div>
  {bars}
  <div style="font-size:13.5px;color:#3a4858;margin-top:12px">Most likely: <b style="color:var(--navy)">{html.escape(top_labels)}</b></div>
</div>

<div class="card" style="border-left:4px solid {acol}">
  <h3 style="color:{acol}">Acute management</h3>
  {redflags}
  <div style="font-weight:600;color:var(--ink);font-size:15.5px;line-height:1.4">{html.escape(ac["headline"])}</div>
  {acute_details}
</div>

<div class="card">
  <h3>Prevention &amp; patient education</h3>
  <div style="font-size:13.5px;color:#3a4858;line-height:1.55">{html.escape(prev["universal"])}</div>
  {flags}
  {blocks}
  <div style="font-size:13px;color:#34506a;background:#f4f8fc;border:1px solid #d8e4f0;border-radius:8px;padding:10px 12px;margin-top:10px;line-height:1.55">{html.escape(prev["workup"])}</div>
  <div style="font-size:11px;color:#94a3b1;margin-top:10px">Sources: {html.escape(prev["cite"])}</div>
</div>""")
