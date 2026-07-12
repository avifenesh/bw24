#!/usr/bin/env python3
"""Build the standalone interactive expert-routing tier explorer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                "rank": int(row["rank"]),
                "expertPct": float(row["expert_percent"]),
                "share": float(row["aggregate_route_share_percent"]),
                "cumulative": float(row["aggregate_cumulative_percent"]),
                "p10": float(row["layer_p10_cumulative_percent"]),
                "median": float(row["layer_median_cumulative_percent"]),
                "p90": float(row["layer_p90_cumulative_percent"]),
            }
            for row in reader
        ]


HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hy3 routing topology · bw24</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #080b12;
      --surface: #0d121d;
      --surface-2: #111827;
      --line: #263041;
      --muted: #8a96a8;
      --text: #edf2f7;
      --nv: #ffcf5c;
      --q3: #65d7c2;
      --q2: #8ea7ff;
      --prune: #687284;
      --curve: #f2f5f8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-width: 320px;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button, input { font: inherit; }
    .shell { min-height: 100svh; display: grid; grid-template-rows: auto 1fr; }
    header {
      display: flex; align-items: baseline; justify-content: space-between; gap: 24px;
      padding: 24px 32px 18px; border-bottom: 1px solid var(--line);
    }
    .brand { display: flex; align-items: baseline; gap: 14px; }
    .brand strong { font-size: 18px; letter-spacing: -.02em; }
    .brand span, .provenance { color: var(--muted); font-size: 12px; }
    main { display: grid; grid-template-columns: minmax(0, 1fr) 330px; min-height: 0; }
    .workspace { padding: 26px 30px 30px; min-width: 0; }
    .workspace-head { display: flex; justify-content: space-between; align-items: end; gap: 22px; margin-bottom: 18px; }
    h1 { font-size: clamp(26px, 3.2vw, 44px); line-height: 1; letter-spacing: -.045em; margin: 0 0 9px; font-weight: 660; }
    .lede { color: var(--muted); margin: 0; max-width: 670px; }
    .coverage { text-align: right; flex: 0 0 auto; }
    .coverage strong { display: block; font-size: 32px; line-height: 1; letter-spacing: -.04em; font-variant-numeric: tabular-nums; }
    .coverage span { color: var(--muted); font-size: 12px; }
    .chart-wrap { position: relative; border-top: 1px solid var(--line); padding-top: 18px; }
    svg { display: block; width: 100%; height: auto; overflow: visible; touch-action: none; }
    .grid { stroke: #202a39; stroke-width: 1; }
    .axis { fill: var(--muted); font-size: 11px; }
    .band { opacity: .105; transition: opacity 160ms ease; }
    .band:hover { opacity: .16; }
    .curve { fill: none; stroke: var(--curve); stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }
    .layer-band { fill: #6abef4; opacity: .09; }
    .median { fill: none; stroke: #78c9f7; stroke-width: 1.5; stroke-dasharray: 5 5; opacity: .7; }
    .density { fill: none; stroke: #a897ff; stroke-width: 2; }
    .density-fill { fill: url(#densityGradient); }
    .handle-line { stroke-width: 1.5; stroke-dasharray: 5 5; cursor: ew-resize; }
    .handle { cursor: ew-resize; filter: drop-shadow(0 2px 6px #0008); }
    .tooltip {
      position: absolute; pointer-events: none; opacity: 0; transform: translate(10px, -8px);
      background: #070a11ee; border: 1px solid #344054; border-radius: 8px; padding: 9px 11px;
      color: var(--text); font-size: 12px; font-variant-numeric: tabular-nums; transition: opacity 80ms ease;
    }
    .tooltip b { display: block; margin-bottom: 3px; }
    .tier-strip { display: grid; grid-template-columns: repeat(4, 1fr); margin-top: 18px; border-top: 1px solid var(--line); }
    .tier { padding: 16px 15px 0 0; min-width: 0; }
    .tier + .tier { padding-left: 15px; border-left: 1px solid var(--line); }
    .tier-label { display: flex; align-items: center; gap: 7px; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }
    .swatch { width: 7px; height: 7px; border-radius: 50%; background: var(--tier); }
    .tier strong { display: block; margin-top: 8px; font-size: 20px; font-variant-numeric: tabular-nums; }
    .tier small { color: var(--muted); }
    aside { border-left: 1px solid var(--line); padding: 28px 24px; background: var(--surface); overflow: auto; }
    aside h2 { font-size: 13px; margin: 0 0 14px; letter-spacing: .02em; }
    .presets { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 28px; }
    .preset { color: var(--muted); background: transparent; border: 1px solid var(--line); border-radius: 999px; padding: 6px 10px; cursor: pointer; transition: 140ms ease; }
    .preset:hover, .preset.active { color: var(--text); border-color: #667085; background: #ffffff08; }
    .control { padding: 17px 0; border-top: 1px solid var(--line); }
    .control-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 11px; }
    .control-title { display: flex; align-items: center; gap: 8px; }
    .control output { color: var(--text); font-variant-numeric: tabular-nums; }
    input[type="range"] { width: 100%; accent-color: var(--accent); cursor: ew-resize; }
    .range-meta { display: flex; justify-content: space-between; color: var(--muted); font-size: 11px; margin-top: 5px; }
    .metrics { margin-top: 27px; border-top: 1px solid var(--line); }
    .metric { display: flex; justify-content: space-between; gap: 20px; padding: 13px 0; border-bottom: 1px solid var(--line); }
    .metric span { color: var(--muted); }
    .metric strong { font-variant-numeric: tabular-nums; text-align: right; }
    .note { color: var(--muted); font-size: 11px; margin: 20px 0 0; }
    @media (max-width: 900px) {
      header { padding-inline: 20px; } .provenance { display: none; }
      main { grid-template-columns: 1fr; } aside { border-left: 0; border-top: 1px solid var(--line); }
      .workspace { padding: 22px 18px; } .workspace-head { align-items: start; } .tier-strip { grid-template-columns: 1fr 1fr; }
      .tier:nth-child(3) { border-left: 0; padding-left: 0; }
    }
    @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
  </style>
</head>
<body>
<div class="shell">
  <header>
    <div class="brand"><strong>bw24</strong><span>routing topology</span></div>
    <div class="provenance">Hy3 · corrected non-REAP trace · 103,274,488 routes</div>
  </header>
  <main>
    <section class="workspace">
      <div class="workspace-head">
        <div><h1>Shape the quantization tiers.</h1><p class="lede">Drag the boundaries across experts ranked independently within each layer. Traffic and storage update from the measured distribution.</p></div>
        <div class="coverage"><strong id="coverageValue">—</strong><span>observed traffic retained</span></div>
      </div>
      <div class="chart-wrap" id="chartWrap">
        <svg id="chart" viewBox="0 0 1020 520" role="img" aria-label="Interactive cumulative expert routing distribution">
          <defs><linearGradient id="densityGradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#a897ff" stop-opacity=".38"/><stop offset="1" stop-color="#a897ff" stop-opacity=".02"/></linearGradient></defs>
          <g id="plot"></g>
        </svg>
        <div class="tooltip" id="tooltip"></div>
      </div>
      <div class="tier-strip" id="tierStrip"></div>
    </section>
    <aside>
      <h2>Presets</h2>
      <div class="presets" id="presets"></div>
      <h2>Tier boundaries</h2>
      <div id="controls"></div>
      <div class="metrics">
        <div class="metric"><span>Logical artifact</span><strong id="logicalSize">—</strong></div>
        <div class="metric"><span>Saved vs plain NVFP4</span><strong id="savedSize">—</strong></div>
        <div class="metric"><span>Pruned traffic</span><strong id="prunedTraffic">—</strong></div>
        <div class="metric"><span>Retained experts</span><strong id="retainedExperts">—</strong></div>
      </div>
      <p class="note">Storage is exact for the current sparse overlay layout: 79 MoE layers plus the shared 24,999,514,624-byte non-expert payload. Quality impact is not predicted by traffic alone.</p>
    </aside>
  </main>
</div>
<script>
const rows = __DATA__;
const TOTAL_EXPERTS = 192, LAYERS = 79;
const COMMON_BYTES = 24999514624;
const PLAIN_LOGICAL = 186035622400;
const BYTES = { q8: 20054016, nv: 10616832, q2: 6193152 };
const tiers = [
  { key:'q8', label:'Q8_0', color:'#ffcf5c' },
  { key:'nv', label:'NVFP4', color:'#65d7c2' },
  { key:'q2', label:'Q2_K', color:'#8ea7ff' },
  { key:'prune', label:'Pruned', color:'#687284' }
];
const presets = [
  { name:'22 / 50 / 85', values:[16,53,126] },
  { name:'90% retained', values:[16,53,142] },
  { name:'95% retained', values:[16,53,160] },
  { name:'No pruning', values:[16,53,192] }
];
let boundaries = [...presets[0].values];
const svg = document.querySelector('#chart'), plot = document.querySelector('#plot');
const W=1020,H=520, left=58,right=20,plotTop=22,mainBottom=342,densityTop=385,densityBottom=485;
const x = rank => left + rank/TOTAL_EXPERTS*(W-left-right);
const y = pct => mainBottom - pct/100*(mainBottom-plotTop);
const yd = share => {
  const lo=Math.log10(.025), hi=Math.log10(3.2);
  return densityBottom-(Math.log10(Math.max(share,.025))-lo)/(hi-lo)*(densityBottom-densityTop);
};
const cumulative = rank => rank <= 0 ? 0 : rows[rank-1].cumulative;
const fmtPct = n => `${n.toFixed(n < 10 ? 2 : 1)}%`;
const fmtGiB = bytes => `${(bytes/2**30).toFixed(2)} GiB`;
const path = (values, mapY) => values.map((v,i)=>`${i?'L':'M'}${x(i+1).toFixed(2)},${mapY(v).toFixed(2)}`).join(' ');
const areaPath = (values, mapY, bottom) => `${path(values,mapY)} L${x(TOTAL_EXPERTS)},${bottom} L${x(1)},${bottom} Z`;
const el = (name, attrs={}) => { const n=document.createElementNS('http://www.w3.org/2000/svg',name); for(const [k,v] of Object.entries(attrs)) n.setAttribute(k,v); return n; };

function drawBase(){
  plot.replaceChildren();
  for(let p=0;p<=100;p+=20){
    plot.append(el('line',{x1:left,y1:y(p),x2:W-right,y2:y(p),class:'grid'}));
    const t=el('text',{x:left-10,y:y(p)+4,'text-anchor':'end',class:'axis'}); t.textContent=`${p}%`; plot.append(t);
  }
  for(let p=0;p<=100;p+=10){
    const xx=left+p/100*(W-left-right); plot.append(el('line',{x1:xx,y1:plotTop,x2:xx,y2:mainBottom,class:'grid'}));
    const t=el('text',{x:xx,y:mainBottom+22,'text-anchor':'middle',class:'axis'}); t.textContent=`${p}%`; plot.append(t);
  }
  const bandTop=path(rows.map(r=>r.p90),y), bandBottom=[...rows].reverse().map((r,i)=>`${i?'L':'L'}${x(TOTAL_EXPERTS-i).toFixed(2)},${y(r.p10).toFixed(2)}`).join(' ');
  plot.append(el('path',{d:`${bandTop} ${bandBottom} Z`,class:'layer-band'}));
  plot.append(el('path',{d:path(rows.map(r=>r.median),y),class:'median'}));
  plot.append(el('path',{d:path(rows.map(r=>r.cumulative),y),class:'curve'}));
  const density=rows.map(r=>r.share); plot.append(el('path',{d:areaPath(density,yd,densityBottom),class:'density-fill'})); plot.append(el('path',{d:path(density,yd),class:'density'}));
  const label=el('text',{x:left,y:densityTop-12,class:'axis'}); label.textContent='ROUTING DENSITY · LOG SCALE'; plot.append(label);
  const overlay=el('rect',{x:left,y:plotTop,width:W-left-right,height:densityBottom-plotTop,fill:'transparent',style:'cursor:crosshair'}); overlay.addEventListener('pointermove',showTooltip); overlay.addEventListener('pointerleave',()=>tooltip.style.opacity=0); plot.append(overlay);
}

function render(){
  drawBase();
  const cuts=[0,...boundaries,TOTAL_EXPERTS];
  tiers.forEach((tier,i)=>{
    const x0=x(cuts[i]), x1=x(cuts[i+1]);
    plot.insertBefore(el('rect',{x:x0,y:plotTop,width:Math.max(0,x1-x0),height:densityBottom-plotTop,fill:tier.color,class:'band'}),plot.firstChild);
  });
  boundaries.forEach((rank,i)=>{
    const tier=tiers[i], xx=x(rank);
    const line=el('line',{x1:xx,y1:plotTop,x2:xx,y2:densityBottom,stroke:tier.color,class:'handle-line','data-index':i});
    const circle=el('circle',{cx:xx,cy:y(cumulative(rank)),r:7,fill:tier.color,class:'handle','data-index':i});
    for(const node of [line,circle]) node.addEventListener('pointerdown',startDrag);
    plot.append(line,circle);
  });
  updateNumbers();
}

function tierRanges(){
  const cuts=[0,...boundaries,TOTAL_EXPERTS];
  return tiers.map((tier,i)=>{ const count=cuts[i+1]-cuts[i], traffic=cumulative(cuts[i+1])-cumulative(cuts[i]); return {...tier,start:cuts[i]+1,end:cuts[i+1],count,traffic}; });
}
function updateNumbers(){
  const ranges=tierRanges(), retained=boundaries[2], coverage=cumulative(retained);
  document.querySelector('#coverageValue').textContent=fmtPct(coverage);
  document.querySelector('#tierStrip').innerHTML=ranges.map(r=>`<div class="tier" style="--tier:${r.color}"><div class="tier-label"><i class="swatch"></i>${r.label}</div><strong>${r.count} <small>· ${(r.count/TOTAL_EXPERTS*100).toFixed(1)}%</small></strong><small>${fmtPct(r.traffic)} of routes</small></div>`).join('');
  const logical=COMMON_BYTES+LAYERS*(ranges[0].count*BYTES.q8+ranges[1].count*BYTES.nv+ranges[2].count*BYTES.q2);
  document.querySelector('#logicalSize').textContent=fmtGiB(logical);
  document.querySelector('#savedSize').textContent=`${fmtGiB(PLAIN_LOGICAL-logical)} · ${((1-logical/PLAIN_LOGICAL)*100).toFixed(1)}%`;
  document.querySelector('#prunedTraffic').textContent=fmtPct(100-coverage);
  document.querySelector('#retainedExperts').textContent=`${retained}/192 · ${(retained/TOTAL_EXPERTS*100).toFixed(1)}%`;
  document.querySelectorAll('input[type=range]').forEach((input,i)=>{ input.value=boundaries[i]; document.querySelector(`#out${i}`).textContent=`${boundaries[i]} · ${(boundaries[i]/TOTAL_EXPERTS*100).toFixed(1)}%`; });
  document.querySelectorAll('.preset').forEach((b,i)=>b.classList.toggle('active',presets[i].values.every((v,j)=>v===boundaries[j])));
}

function setBoundary(i,value){
  const min=i===0?1:boundaries[i-1]+1, max=i===2?TOTAL_EXPERTS:boundaries[i+1]-1;
  boundaries[i]=Math.max(min,Math.min(max,Math.round(value))); render();
}
let dragIndex=null;
function startDrag(e){ dragIndex=Number(e.currentTarget.dataset.index); e.currentTarget.setPointerCapture(e.pointerId); e.currentTarget.addEventListener('pointermove',drag); e.currentTarget.addEventListener('pointerup',endDrag,{once:true}); }
function drag(e){ const pt=svg.createSVGPoint(); pt.x=e.clientX; pt.y=e.clientY; const local=pt.matrixTransform(svg.getScreenCTM().inverse()); setBoundary(dragIndex,(local.x-left)/(W-left-right)*TOTAL_EXPERTS); }
function endDrag(e){ e.currentTarget.removeEventListener('pointermove',drag); dragIndex=null; }
const tooltip=document.querySelector('#tooltip'), wrap=document.querySelector('#chartWrap');
function showTooltip(e){
  const rect=svg.getBoundingClientRect(), localX=(e.clientX-rect.left)/rect.width*W; const rank=Math.max(1,Math.min(TOTAL_EXPERTS,Math.round((localX-left)/(W-left-right)*TOTAL_EXPERTS))); const row=rows[rank-1];
  tooltip.innerHTML=`<b>Rank ${rank} · ${row.expertPct.toFixed(1)}% of experts</b>${row.cumulative.toFixed(2)}% cumulative traffic<br>${row.share.toFixed(4)}% at this rank`;
  tooltip.style.left=`${Math.min(e.clientX-wrap.getBoundingClientRect().left,wrap.clientWidth-190)}px`; tooltip.style.top=`${e.clientY-wrap.getBoundingClientRect().top}px`; tooltip.style.opacity=1;
}

document.querySelector('#presets').innerHTML=presets.map((p,i)=>`<button class="preset" data-index="${i}">${p.name}</button>`).join('');
document.querySelectorAll('.preset').forEach(b=>b.addEventListener('click',()=>{ boundaries=[...presets[Number(b.dataset.index)].values]; render(); }));
document.querySelector('#controls').innerHTML=tiers.slice(0,3).map((tier,i)=>`<div class="control" style="--accent:${tier.color}"><div class="control-head"><div class="control-title"><i class="swatch" style="--tier:${tier.color}"></i><span>${tier.label} ends</span></div><output id="out${i}"></output></div><input type="range" min="1" max="192" step="1" aria-label="${tier.label} upper rank"><div class="range-meta"><span>hotter</span><span>colder</span></div></div>`).join('');
document.querySelectorAll('input[type=range]').forEach((input,i)=>input.addEventListener('input',e=>setBoundary(i,Number(e.target.value))));
render();
</script>
</body>
</html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.input)
    if len(rows) != 192:
        raise ValueError(f"expected 192 ranked experts, found {len(rows)}")
    args.output.write_text(HTML.replace("__DATA__", json.dumps(rows, separators=(",", ":"))))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
