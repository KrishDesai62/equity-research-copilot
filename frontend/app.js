'use strict';
const $ = (s, root = document) => root.querySelector(s);
const main = $('#main');
const currentDate=()=>new Date().toLocaleDateString('en-CA',{timeZone:'America/Chicago'});
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const safeUrl = value => { try { const u = new URL(value); return u.protocol === 'https:' && /(^|\.)sec\.gov$/.test(u.hostname) ? u.href : '#'; } catch { return '#'; } };
const date = value => value ? new Date(value + 'T12:00:00').toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric'}) : 'Unavailable';
const shortDate = value => value ? new Date(value+'T12:00:00').toLocaleDateString('en-US',{month:'short',year:'2-digit'}) : '';
const money = value => value == null ? '—' : new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',notation:'compact',maximumFractionDigits:1}).format(value);
const pct = value => value == null ? '—' : (value*100).toFixed(1)+'%';
const num = value => value == null ? '—' : new Intl.NumberFormat('en-US').format(value);
const sectionName = section => ({risk_factors:'Risk factors',mda:'Management discussion',md_and_a:'Management discussion',management_discussion:'Management discussion'}[section] || String(section||'Filing evidence').replaceAll('_',' '));
function readStorage(key, fallback) { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } }
const state = {catalog:null, data:null, ticker:'AAPL', view:'company', tab:'overview', asOf:'2026-10-16', chart:'Revenues_ttm', query:'', section:'all', form:'all', stars:readStorage('ledger.watchlist.v1',['AAPL','MSFT','NVDA','AMZN','GOOGL']), notes:readStorage('ledger.notes.v1',{}), request:0};
state.apiBase = null;
state.importsEnabled = false;
state.researchEnabled = false;
state.importJob = null;
let importTimer;
const cache = new Map();
let toastTimer;
function toast(text) { $('#toast').textContent=text; $('#toast').classList.add('visible'); clearTimeout(toastTimer); toastTimer=setTimeout(()=>$('#toast').classList.remove('visible'),2600); }
let accountSaveTimer;
function store(key,value) {
 if(window.ledgerAccount?.required){
   if(!window.ledgerAccount.user){toast('Sign in to save research across devices.');return false;}
   clearTimeout(accountSaveTimer);
   accountSaveTimer=setTimeout(async()=>{
    try {const r=await fetch('/api/me',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({watchlist:state.stars,notes:state.notes})});if(!r.ok)throw Error();}
    catch{toast('Could not save research. Export your notes and retry.');}
   },500);return true;
 }
 try { localStorage.setItem(key,JSON.stringify(value)); return true; } catch { toast('Browser storage is unavailable. Export your note to keep it.'); return false; } }
function routeCompany(ticker, tab='overview') { location.hash=`company/${encodeURIComponent(ticker)}/${tab}`; }
function pageTop(title, label='SEC FILINGS & FINANCIALS') { return `<div class="page-top"><div><div class="eyebrow">${label}</div><h1>${title}</h1></div><label class="as-of">Evidence available by<input id="as-of" aria-label="Evidence cutoff date" type="date" value="${state.asOf}" max="${currentDate()}"></label></div>`; }
function sourceLink(url,label,cls='') { return `<a class="${cls}" href="${esc(safeUrl(url))}" target="_blank" rel="noopener noreferrer">${label}</a>`; }
function visibleHistory() { return state.data.history.filter(r=>r.filed<=state.asOf); }
function visiblePassages() { return state.data.passages.filter(r=>r.filing_date<=state.asOf); }
function visibleFilings() { return state.data.filings.filter(r=>r.filing_date<=state.asOf).sort((a,b)=>b.filing_date.localeCompare(a.filing_date)); }
function latest() { return visibleHistory().at(-1) || null; }
function sidebar() {
  document.querySelectorAll('.nav-item').forEach(b=>b.classList.toggle('active',b.dataset.view===state.view));
  $('#watch-count').textContent=state.stars.filter(t=>state.catalog.companies.some(c=>c.ticker===t)).length;
  const starred=state.stars.map(t=>state.catalog.companies.find(c=>c.ticker===t)).filter(Boolean);
  $('#following').innerHTML=starred.length ? starred.slice(0,9).map(c=>`<button class="follow-item ${state.view==='company'&&state.ticker===c.ticker?'selected':''}" data-company="${esc(c.ticker)}"><i class="follow-dot"></i>${esc(c.ticker)}<span class="follow-name">${esc(c.name.replace(/ Inc\.?| Corp\.?| Corporation/gi,''))}</span></button>`).join('') : '<div class="small-note">Follow a company to keep it here.</div>';
  $('#breadcrumb').textContent=({company:'Company research',watchlist:'Watchlist',notes:'Research notes',methodology:'About the data',account:'Account',notifications:'Filing alerts',architecture:'How Ledger works'})[state.view];
}
function tabs() { return `<div class="tabs" role="tablist" aria-label="Company research views">${[['overview','Overview'],['evidence','Filing evidence'],['filings','SEC filings'],['research','Combined research'],['risk','Bankruptcy model'],['changes','Filing changes'],['market','Market & news'],['notes','Notes']].map(([id,label])=>`<button class="tab ${state.tab===id?'active':''}" role="tab" aria-selected="${state.tab===id}" data-tab="${id}">${label}${id==='filings'?`<small>${visibleFilings().length}</small>`:''}</button>`).join('')}</div>`; }
function companyHeader() {
  const c=state.data.company; const l=latest(); const saved=state.stars.includes(c.ticker);
  return `${pageTop('Company research')}<div class="company-heading"><div class="company-monogram" aria-hidden="true">${esc(c.name[0])}</div><div><div class="company-title"><h2>${esc(c.name)}</h2><span class="ticker-tag">${esc(c.ticker)}</span></div><div class="company-subtitle"><span>CIK ${esc(c.cik)}</span><span class="separator">·</span><span>${l?`Financials filed ${date(l.filed)}`:'No financials by this date'}</span></div></div><div class="heading-actions"><button class="button ${saved?'saved':''}" data-action="star">${saved?'★ Following':'☆ Follow company'}</button>${state.importsEnabled?'<button class="button" data-action="refresh-company">Refresh SEC data ↻</button><button class="button" data-action="import-history">Import more history</button>':''}<button class="button" data-action="export">Export brief <span>↗</span></button></div></div>${tabs()}`;
}
function stat(label,value,detail,cls='') { return `<div class="stat"><div class="stat-label">${label}<span>↗</span></div><div class="stat-value">${value}</div><div class="stat-detail ${cls}">${detail}</div></div>`; }
function stats() {
 const l=latest();const growth=l?.revenue_growth_yoy;
 return `<div class="stats">${stat('Revenue · trailing 12 months',money(l?.Revenues_ttm),growth==null?'Year-over-year growth unavailable':`${growth>=0?'↗':'↘'} ${pct(growth)} year over year`,growth!=null?(growth>=0?'positive':'negative'):'')}${stat('Operating income · TTM',money(l?.OperatingIncomeLoss_ttm),l?.Revenues_ttm&&l?.OperatingIncomeLoss_ttm!=null?`${pct(l.OperatingIncomeLoss_ttm/l.Revenues_ttm)} TTM operating margin`:'Reported operating income')}${stat('Cash & equivalents',money(l?.Cash),'At the reported period end')}${stat('Current ratio',l?.current_ratio==null?'—':l.current_ratio.toFixed(2)+'×','Current assets / current liabilities')}</div>`;
}
function chart() {
 const field=state.chart; const rows=visibleHistory().filter(r=>r[field]!=null).slice(-8); const name=field==='Revenues_ttm'?'Revenue':'Operating cash flow';
 let graphic='<div class="chart-empty">Not enough financial history for this date.</div>';
 if(rows.length>1){
  const W=640,H=205,L=61,R=20,T=17,B=35;const values=rows.map(r=>r[field]/1e9); const low=Math.min(0,...values), high=Math.max(...values,1);const pad=(high-low)*.15; const min=low<0?low-pad:0,max=high+pad;
  const x=i=>L+i*(W-L-R)/(rows.length-1);const y=v=>T+(max-v)/(max-min)*(H-T-B);
  const points=values.map((v,i)=>`${x(i)},${y(v)}`).join(' '); const base=y(Math.max(0,min));
  graphic=`<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${name}, trailing twelve months, in USD billions, from ${esc(rows[0].period)} to ${esc(rows.at(-1).period)}"><defs><linearGradient id="chart-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#2677cb" stop-opacity=".22"/><stop offset="100%" stop-color="#2677cb" stop-opacity=".015"/></linearGradient></defs>${[0,1,2,3].map(i=>{const v=min+(max-min)*i/3; return `<line class="grid" x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L-12}" y="${y(v)+3}" text-anchor="end">$${v.toFixed(0)}B</text>`;}).join('')}<polygon points="${x(0)},${base} ${points} ${x(rows.length-1)},${base}" fill="url(#chart-fill)"/><polyline points="${points}" fill="none" stroke="#2677cb" stroke-width="2.3" stroke-linejoin="round"/>${rows.map((r,i)=>`<circle class="chart-point" cx="${x(i)}" cy="${y(values[i])}" r="3.6"><title>${date(r.period)}: ${money(r[field])}. Filed ${date(r.filed)}.</title></circle>${(rows.length<=6||i%2===0||i===rows.length-1)?`<text x="${x(i)}" y="${H-11}" text-anchor="middle">${shortDate(r.period)}</text>`:''}`).join('')}</svg>`;
 }
 return `<section class="panel"><div class="panel-header"><div><h3 class="panel-title">Financial trends</h3><div class="panel-caption">Trailing twelve months · USD billions · period end</div></div><div class="segmented" aria-label="Chart metric"><button data-chart="Revenues_ttm" class="${field==='Revenues_ttm'?'active':''}">Revenue</button><button data-chart="OperatingCashFlow_ttm" class="${field==='OperatingCashFlow_ttm'?'active':''}">Cash flow</button></div></div><div class="chart-wrap">${graphic}</div><div class="chart-source"><span class="legend"><i></i>${name}, TTM</span>${latest()?sourceLink(latest().source_url,'Source: SEC financial statements ↗'):'No source by selected date'}</div></section>`;
}
function financials() {
 const history=visibleHistory();const l=history.at(-1); const prev=history.length>1?history.at(-2):null;
 const metrics=[['Revenue · TTM','Revenues_ttm'],['Operating income · TTM','OperatingIncomeLoss_ttm'],['Net income · TTM','NetIncomeLoss_ttm'],['Operating cash flow · TTM','OperatingCashFlow_ttm'],['Total assets','Assets'],['Total liabilities','Liabilities']];
 return `<section class="panel"><div class="panel-header"><h3 class="panel-title">Financial snapshot</h3><span class="chart-header-detail">USD · reported values</span></div><div class="table-wrap"><table><thead><tr><th>Metric</th><th class="numeric">${l?esc(l.period):'Current'}</th><th class="numeric">${prev?esc(prev.period):'Previous'}</th></tr></thead><tbody>${metrics.map(([label,key])=>`<tr><td class="table-name">${label}</td><td class="numeric">${money(l?.[key])}</td><td class="numeric muted">${money(prev?.[key])}</td></tr>`).join('')}</tbody></table></div><div class="source-line">${l?`Period ended ${date(l.period)}. ${sourceLink(l.source_url,'Read source filing ↗')}`:'No financial statement is available by this cutoff.'}</div></section>`;
}
function filingTable(full=false) {
 let rows=visibleFilings();if(!full)rows=rows.slice(0,3);
 return `<section class="panel"><div class="panel-header"><div><h3 class="panel-title">${full?'Indexed SEC filings':'SEC reports'}</h3>${full?'<div class="panel-caption">Reports with indexed passages in this research snapshot</div>':''}</div>${!full?'<button class="text-button" data-tab="filings">All filings ↗</button>':''}</div><div class="table-wrap"><table><thead><tr><th>Document</th><th>Filed</th><th class="numeric">Source</th></tr></thead><tbody>${rows.map(f=>`<tr><td>${sourceLink(f.source_url,`<span class="filing-link"><span class="file-icon">▤</span><strong>Form ${esc(f.filing_type)}</strong></span>`)}</td><td class="muted">${date(f.filing_date)}</td><td class="numeric">${sourceLink(f.source_url,'SEC ↗','muted')}</td></tr>`).join('')||'<tr><td colspan="3">No indexed reports by the selected date.</td></tr>'}</tbody></table></div></section>`;
}
function researchDesk() {
 return `<section class="panel research-desk"><div class="desk-head"><div class="desk-label"><i></i> FILING SEARCH</div><h3 class="panel-title">Search company filings</h3><p>Search disclosures with links to the original filings.</p></div><div class="desk-body"><form id="question-form"><label class="question-label" for="question">Explore ${esc(state.ticker)}’s filings</label><div class="question-box"><textarea id="question" placeholder="What does the filing say about liquidity?" maxlength="500" required></textarea><div class="question-bottom"><span>${state.apiBase===null?'Search indexed filing passages':'Connected filing retrieval'}</span><button type="submit" class="question-submit" aria-label="Search filing evidence">↗</button></div></div></form><div class="suggestions"><button class="chip" data-query="liquidity cash flow">Liquidity</button><button class="chip" data-query="competition risk">Competition</button><button class="chip" data-query="revenue growth">Growth</button></div><div id="desk-results" class="inline-results" aria-live="polite"></div><div class="desk-foot">Direct excerpts. Original sources.<br>Explore the <button class="text-button" data-tab="evidence">full evidence library ↗</button></div></div></section>`;
}
function coveragePanel() {
 const filings=visibleFilings();const passages=visiblePassages();
 return `<section class="panel"><div class="panel-header"><h3 class="panel-title">Available sources</h3><span class="badge">SEC</span></div><div class="coverage-list"><div class="coverage-row"><span>Indexed reports</span><strong>${filings.length}</strong></div><div class="coverage-row"><span>Searchable passages</span><strong>${num(passages.length)}</strong></div><div class="coverage-row"><span>Latest indexed filing</span><strong>${filings.length?date(filings[0].filing_date):'Unavailable'}</strong></div></div><div class="small-note">Coverage reflects this saved research snapshot.</div></section>`;
}
function notePanel(full=false) {
 const saved=state.notes[state.ticker];return `<section class="panel note-panel"><div class="panel-header"><h3 class="panel-title">${full?'Your research notes':'My notes'}</h3>${!full?'<button class="text-button" data-tab="notes">Open ↗</button>':''}</div>${full?`<label class="full-note-label" for="note-editor">${esc(state.ticker)} · ${window.ledgerAccount?.user?'saved to your account':'saved in this browser'}</label><textarea id="note-editor" class="note-editor" placeholder="What’s worth following up on? Capture your thesis, questions, and source references." style="min-height:260px">${esc(saved?.text||'')}</textarea><div class="note-actions"><button class="button primary" data-action="save-note">Save note</button><button class="button" data-action="download-note">Download .txt</button><small>${saved?`Last saved ${date(saved.date)}`:'Private to this browser'}</small></div>`:`<p class="note-preview">${saved?.text?esc(saved.text.slice(0,220))+(saved.text.length>220?'…':''):'Save your thesis, questions, and filing references.'}</p>`}</section>`;
}
const stopWords=new Set('a an the and or to of in on for with is are was were do does what how why about say says filing filings company their its from by at as it this that can has have be'.split(' '));
function tokens(text) {return (text.toLowerCase().match(/[a-z0-9]{2,}/g)||[]).filter(t=>!stopWords.has(t));}
function searchPassages(query, {section='all',form='all',limit=12}={}) {
 const terms=[...new Set(tokens(query))];let rows=visiblePassages().filter(p=>(section==='all'||p.section===section)&&(form==='all'||p.filing_type===form));
 if(!terms.length)return query.trim()?[]:rows.slice(0,limit);
 const scored=rows.map(p=>{const text=p.text.toLowerCase();let matched=0;let score=0;for(const word of terms){if(text.includes(word)){matched++;score+=1+Math.min((text.match(new RegExp(word.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'),'g'))||[]).length,8)*.1;}}return {p,score:score/Math.sqrt(Math.max(text.length/1000,1)),matched};}).filter(r=>r.matched>=Math.min(terms.length,2));
 return scored.sort((a,b)=>b.score-a.score||b.p.filing_date.localeCompare(a.p.filing_date)).slice(0,limit).map(r=>r.p);
}
function snippet(p,query,max=650) {
 const terms=tokens(query); const lower=p.text.toLowerCase();const matches=terms.map(t=>lower.indexOf(t)).filter(i=>i>=0);const start=matches.length?Math.max(0,Math.min(...matches)-100):0;
 return (start?'…':'')+p.text.slice(start,start+max)+(p.text.length>start+max?'…':'');
}
function evidenceCard(p,query,index,compact=false) {
 const text=snippet(p,query,compact?360:750);
 if(compact)return `<div class="inline-result"><div class="evidence-card-top"><span class="badge">${esc(p.filing_type)}</span><span>${date(p.filing_date)}</span></div><p>${esc(text)}</p>${sourceLink(p.source_url,`${esc(sectionName(p.section))} · Read filing ↗`)}</div>`;
 return `<article class="evidence-card"><div class="evidence-card-top"><span><span class="badge">${esc(p.filing_type)}</span> &nbsp; ${date(p.filing_date)}</span><span>Source ${String(index+1).padStart(2,'0')}</span></div><h3>${esc(sectionName(p.section))}</h3><p>${esc(text)}</p><details><summary class="text-button">Read full passage</summary><p>${esc(p.text)}</p></details><div class="evidence-card-foot"><span>${esc(p.accession_number)}</span>${sourceLink(p.source_url,'Open SEC filing ↗')}</div></article>`;
}
function evidenceView() {
 const sections=[...new Set(visiblePassages().map(p=>p.section))];const forms=[...new Set(visiblePassages().map(p=>p.filing_type))];
 return `<div class="linked-section"><div><h2>The evidence library</h2><p class="panel-caption">Search the words in ${esc(state.ticker)}’s indexed filings. Results are excerpts, with original sources.</p></div></div><form id="evidence-form" class="evidence-toolbar"><input id="evidence-query" type="search" aria-label="Search filing passages" placeholder="Search liquidity, debt, competition…" value="${esc(state.query)}" maxlength="500"><select id="section-filter" aria-label="Filter by filing section"><option value="all">All sections</option>${sections.map(s=>`<option value="${esc(s)}" ${state.section===s?'selected':''}>${esc(sectionName(s))}</option>`).join('')}</select><select id="form-filter" aria-label="Filter by filing form"><option value="all">All forms</option>${forms.map(f=>`<option ${state.form===f?'selected':''}>${esc(f)}</option>`).join('')}</select><button class="button primary" type="submit">Search ↗</button></form><div id="evidence-output" aria-live="polite">${evidenceOutput()}</div>`;
}
function evidenceOutput() {
 const results=searchPassages(state.query,{section:state.section,form:state.form});
 if(!results.length)return '<div class="empty-state"><h3>No matching evidence.</h3>Try a different phrase, section, or date. No answer is generated when sources do not match.</div>';
 return `<p class="evidence-count">${state.query?`Top ${results.length} matching passages`:`Showing ${results.length} of ${num(visiblePassages().length)} passages`} · ${esc(state.ticker)} · through ${date(state.asOf)}</p><div class="evidence-results">${results.map((p,i)=>evidenceCard(p,state.query,i)).join('')}</div>`;
}
function importNotice(){const info=state.data?.import_info;if(!info)return '';return `<details class="import-notice"><summary>Imported ${date(info.as_of)} · ${state.data.company.filing_count} selected SEC reports</summary><p>${esc(info.selection)}</p>${info.warnings.map(w=>`<p>${esc(w)}</p>`).join('')}</details>`;}
function renderCompany(){
 if(!state.data){main.innerHTML=pageTop('Company not covered')+'<div class="empty-state"><h3>No indexed company found.</h3>Use company search to explore the covered companies.</div>';return;}
 let content;if(state.tab==='overview')content=`${stats()}<div class="workspace-grid"><div class="main-column">${chart()}${financials()}${filingTable()}</div><aside class="right-column" aria-label="Research tools">${researchDesk()}${coveragePanel()}${notePanel()}</aside></div>`;
 else if(state.tab==='evidence')content=evidenceView();else if(state.tab==='filings')content=filingTable(true);else if(['research','risk','changes','market'].includes(state.tab))content='<div id="analysis-output" aria-live="polite"><div class="loading"><span class="loader"></span>Checking company sources…</div></div>';else content=`${notePanel(true)}<p class="note-disclaimer">${window.ledgerAccount?.user?'Notes are saved to your account. Use Account settings to export your data.':'Notes stay in this browser. Download a copy to keep them.'}</p>`;
 main.innerHTML=companyHeader()+importNotice()+content;
 if(['research','risk','changes','market'].includes(state.tab))loadAnalysis();
}
function renderWatchlist() {
 const companies=state.catalog.companies.filter(c=>state.stars.includes(c.ticker));
 main.innerHTML=pageTop('Your watchlist','A FEW COMPANIES. A CLOSER LOOK.')+`<div class="universe-summary"><span><b>${companies.length}</b> companies followed</span><span>Reported financials · USD</span><button class="text-button" id="browse-companies">Find another company ↗</button></div>`+(companies.length?`<section class="panel"><div class="table-wrap"><table class="watchlist-table"><thead><tr><th>Company</th><th class="numeric">Revenue · TTM</th><th class="numeric">Cash</th><th class="numeric">Indexed passages</th><th></th></tr></thead><tbody>${companies.map(c=>{const l=c.latest?.filed<=state.asOf?c.latest:null;return `<tr><td><button class="row-button company-cell" data-company="${esc(c.ticker)}"><span class="mini-monogram">${esc(c.name[0])}</span><span><strong>${esc(c.ticker)} ↗</strong><small>${esc(c.name)}</small></span></button></td><td class="numeric">${money(l?.Revenues_ttm)}</td><td class="numeric">${money(l?.Cash)}</td><td class="numeric muted">${num(c.passage_count)}</td><td class="numeric"><button class="star-button on" data-unstar="${esc(c.ticker)}" aria-label="Unfollow ${esc(c.ticker)}">★</button></td></tr>`;}).join('')}</tbody></table></div></section><p class="end-note">Financials are shown only when their latest saved filing precedes the cutoff. Passage counts reflect the full snapshot.</p>`:'<div class="empty-state"><h3>Start with a company.</h3>Find a company and follow it to build your watchlist.</div>');
}
function renderNotes() {
 const entries=Object.entries(state.notes).filter(([,n])=>n.text?.trim());main.innerHTML=pageTop('Research notes','SAVED RESEARCH')+(entries.length?`<div class="notes-grid">${entries.map(([ticker,n])=>`<article class="note-card"><span class="eyebrow">${esc(ticker)} · ${date(n.date)}</span><h3>${esc(state.catalog.companies.find(c=>c.ticker===ticker)?.name||ticker)}</h3><p>${esc(n.text)}</p><button class="text-button" data-note-company="${esc(ticker)}">Continue writing ↗</button></article>`).join('')}</div>`:'<div class="empty-state"><h3>A space for your thinking.</h3>Open a company’s Notes tab to record a thesis, questions, and source references.</div>');
}
function renderMethodology() {
 const t=state.catalog.totals;main.innerHTML=pageTop('Behind the numbers','METHOD & COVERAGE')+`<div class="methodology"><section class="panel"><h2>Filings first. Conclusions second.</h2><p>Ledger is a research desk for exploring public SEC filings. Financial data comes from EDGAR/XBRL; excerpts link back to their original filing. The initial research collection uses a saved public-data snapshot. With the Python server running, enter another ticker in company search to import selected public SEC reports and financials. Imports are saved on the server; a page refresh does not refresh existing reports.</p><dl><div><dt>Covered companies</dt><dd>${t.companies}</dd></div><div><dt>Indexed SEC reports</dt><dd>${t.filings}</dd></div><div><dt>Searchable passages</dt><dd>${num(t.passages)}</dd></div><div><dt>Snapshot date</dt><dd>${date(state.catalog.snapshot_date)}</dd></div></dl></section><section class="panel"><h2>Read the evidence in context.</h2><p>In this static edition, passage search matches keywords within the selected company, filing type, section, and cutoff date. It returns cited excerpts rather than generated answers. ${state.apiBase===null?'The question box also uses keyword search in this static edition. Run the included Python web server to connect it to the project’s existing retrieval pipeline.':'The question box is connected to the project’s existing Python retrieval pipeline; the evidence library remains a browser-side keyword search.'}</p><p>Financial trends use the last available period snapshots and reported trailing-twelve-month values. A dash means a value is unavailable. Source-fidelity checks verify that excerpts correspond to their filing sources; they do not establish answer accuracy.</p><p>Bankruptcy models remain research artifacts. This interface does not display an unreleased risk probability or turn a missing model result into a favorable rating. ${window.ledgerAccount?.required?'Signed-in notes and watchlists are stored privately in PostgreSQL.':'Notes and followed companies are saved locally in this browser.'}</p><p><a href="https://github.com/KrishDesai62/equity-research-copilot" target="_blank" rel="noopener noreferrer">Explore the project and methodology ↗</a></p></section></div>`;
}
function render() {
 sidebar();document.title=`Ledger — ${state.view==='company'?state.data?.company.name||'Company research':({watchlist:'Watchlist',notes:'Research notes',methodology:'About the data',account:'Account',notifications:'Filing alerts',architecture:'How Ledger works'})[state.view]}`;
 if(state.view==='architecture'){window.renderArchitecture();return;}
 if(['account','notifications'].includes(state.view)){window.renderAccount(state.view);return;}
 if(state.view==='company')renderCompany();else if(state.view==='watchlist')renderWatchlist();else if(state.view==='notes')renderNotes();else renderMethodology();
}
async function route() {
 if(!state.catalog)return;const parts=location.hash.slice(1).split('/');const view=['company','watchlist','notes','methodology','account','notifications','architecture'].includes(parts[0])?parts[0]:'company';const oldTicker=state.ticker;state.view=view;state.ticker=decodeURIComponent(parts[1]||state.ticker);state.tab=['overview','evidence','filings','research','risk','changes','market','notes'].includes(parts[2])?parts[2]:'overview';
 $('#sidebar').classList.remove('open');$('#mobile-menu').setAttribute('aria-expanded','false');
 if(view==='company'){
   if(!state.catalog.companies.some(c=>c.ticker===state.ticker)){state.data=null;main.innerHTML=pageTop('Company not covered')+'<div class="empty-state"><h3>No indexed company found.</h3>Use company search to explore saved companies.</div>';sidebar();return;}
   if(oldTicker!==state.ticker){state.query='';state.section='all';state.form='all';}
   const request=++state.request;
   try {if(!cache.has(state.ticker)){main.innerHTML='<div class="loading"><span class="loader"></span>Opening company filings…</div>';const response=await fetch(`./data/companies/${encodeURIComponent(state.ticker)}.json`);if(!response.ok)throw Error('Company data unavailable');cache.set(state.ticker,await response.json());}if(request!==state.request||state.view!=='company')return;state.data=cache.get(state.ticker);}catch{main.innerHTML='<div class="empty-state"><h3>Could not open these filings.</h3>Check your connection and reload the page.</div>';return;}
 } else {state.request++;}
 render();window.scrollTo({top:0,behavior:'instant'});
}
function showSearch() {$('#company-dialog').showModal();$('#company-search').value='';renderSearch('');$('#company-search').focus();}
function renderSearch(query) {
 const q=query.trim().toLowerCase();const rows=state.catalog.companies.filter(c=>c.ticker.toLowerCase().includes(q)||c.name.toLowerCase().includes(q));$('#search-meta').textContent=`${rows.length} ${rows.length===1?'company':'companies'} ${q?'found':'in the research universe'}`;
 $('#company-results').innerHTML=rows.map(c=>`<button class="company-result" data-company="${esc(c.ticker)}"><span class="mini-monogram">${esc(c.name[0])}</span><span><strong>${esc(c.ticker)}</strong><small>${esc(c.name)}</small></span><span class="result-arrow">↗</span></button>`).join('')||'<div class="empty-state">No saved company matches that search.</div>';
 const symbol=query.trim().toUpperCase().replaceAll('.','-');
 const alreadySaved=state.catalog.companies.some(c=>c.ticker.replaceAll('.','-')===symbol);
 const box=state.importJob?importProgress(state.importJob):(!alreadySaved&&/^[A-Z][A-Z0-9-]{0,19}$/.test(symbol)?`<div class="import-box"><h3>Research ${esc(symbol)}.</h3><p>${state.importsEnabled?'Import public SEC reports and financials, then search them here. The first import may take a few minutes.':'Importing new companies requires the Python server. This static site includes saved companies only.'}</p>${state.importsEnabled?`<button class="button primary" data-import-ticker="${esc(symbol)}">Import ${esc(symbol)} from SEC ↗</button>`:''}</div>`:'');
 $('#company-results').insertAdjacentHTML('beforeend',box);
}
function toggleStar(ticker) {state.stars=state.stars.includes(ticker)?state.stars.filter(t=>t!==ticker):[...state.stars,ticker];store('ledger.watchlist.v1',state.stars);render();toast(state.stars.includes(ticker)?`${ticker} added to your watchlist`:`${ticker} removed from your watchlist`);}
function download(name,text) {const url=URL.createObjectURL(new Blob([text],{type:'text/plain;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function exportBrief(){const l=latest();const c=state.data.company;download(`ledger-${state.ticker}-brief.txt`,`${c.name} (${c.ticker})\nLedger research brief · evidence through ${state.asOf}\n\nFINANCIAL SNAPSHOT\nPeriod end: ${l?.period||'Unavailable'}\nRevenue (TTM): ${money(l?.Revenues_ttm)}\nOperating income (TTM): ${money(l?.OperatingIncomeLoss_ttm)}\nCash: ${money(l?.Cash)}\nFinancial source: ${l?.source_url||'Unavailable'}\n\nINDEXED FILINGS\n${visibleFilings().map(f=>`${f.filing_type} · ${f.filing_date}\n${f.source_url}`).join('\n\n')}\n\nYOUR NOTES\n${state.notes[state.ticker]?.text||'No notes saved.'}\n\nSaved public SEC snapshot. Research only; no investment recommendation.\n`);toast('Research brief downloaded');}
document.addEventListener('click',event=>{
 const target=event.target.closest('button');if(!target)return;
 if(target.dataset.importTicker){startImport(target.dataset.importTicker);return;}
 if(target.dataset.action==='dismiss-import'){state.importJob=null;renderSearch($('#company-search').value);return;}
 if(target.dataset.company){$('#company-dialog').close();routeCompany(target.dataset.company);return;}
 if(target.dataset.view){location.hash=target.dataset.view==='company'?`company/${state.ticker}`:target.dataset.view;return;}
 if(target.dataset.tab){routeCompany(state.ticker,target.dataset.tab);return;}
 if(target.dataset.chart){state.chart=target.dataset.chart;renderCompany();return;}
 if(target.dataset.unstar){toggleStar(target.dataset.unstar);return;}
 if(target.dataset.noteCompany){routeCompany(target.dataset.noteCompany,'notes');return;}
 if(target.dataset.query){const question=$('#question');if(question)question.value=target.dataset.query;$('#question-form')?.requestSubmit();return;}
 if(['global-search','add-company','browse-companies'].includes(target.id)){showSearch();return;}
 if(target.id==='close-dialog'){$('#company-dialog').close();return;}
 if(target.id==='mobile-menu'){const open=$('#sidebar').classList.toggle('open');target.setAttribute('aria-expanded',String(open));return;}
 if(target.dataset.action==='import-history'){startImport(state.ticker,true,true);showSearch();return;}
 if(target.dataset.action==='refresh-company'){startImport(state.ticker,true);showSearch();return;}
 if(target.dataset.sourceId){document.getElementById('research-source-'+target.dataset.sourceId)?.scrollIntoView({behavior:'smooth',block:'center'});return;}
 if(target.dataset.action==='retry-analysis'){loadAnalysis();return;}
 if(target.dataset.action==='star')toggleStar(state.ticker);
 if(target.dataset.action==='export')exportBrief();
 if(target.dataset.action==='save-note'){const text=$('#note-editor').value;if(store('ledger.notes.v1',{...state.notes,[state.ticker]:{text,date:new Date().toLocaleDateString('en-CA',{timeZone:'America/Chicago'})}})){state.notes[state.ticker]={text,date:new Date().toLocaleDateString('en-CA',{timeZone:'America/Chicago'})};toast(window.ledgerAccount?.user?'Note queued for account saving':'Note saved in this browser');}}
 if(target.dataset.action==='download-note')download(`ledger-${state.ticker}-notes.txt`,$('#note-editor')?.value||state.notes[state.ticker]?.text||'');
});
document.addEventListener('submit',async event=>{
 if(event.target.id==='question-form'){
  event.preventDefault(); const query=$('#question').value.trim(); if(!query)return;
  const ticker=state.ticker, cutoff=state.asOf; const output=$('#desk-results'); const submit=$('.question-submit'); submit.disabled=true;
  output.innerHTML='<div class="form-status">Looking through the filings…</div>';
  try {
   let results;
   if(state.apiBase!==null){
    const response=await fetch(`${state.apiBase}/api/answer`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:query,ticker,as_of:cutoff})});
    const payload=await response.json(); window.__lastSynthesis=payload.synthesis; if(!response.ok)throw Error(response.status===401?'Sign in to search company filings.':response.status===429?'Request limit reached. Please wait a minute and retry.':'Retrieval is temporarily unavailable.');
    results=payload.citations.filter(c=>c.filing_date<=cutoff).map(c=>({text:c.excerpt,section:c.section,filing_date:c.filing_date,source_url:c.source_url,filing_type:'SEC',accession_number:c.accession_number}));
   } else results=searchPassages(query,{limit:3});
   if(state.ticker!==ticker||state.asOf!==cutoff||!output.isConnected)return;
   output.innerHTML=results.length?`<div class="form-status">${results.length} cited passages found</div>${synthesisOutput(window.__lastSynthesis)}${results.map((p,i)=>evidenceCard(p,query,i,true)).join('')}`:'<div class="form-status">No matching evidence by this date. Try a different phrase.</div>';
  }catch(error){if(output.isConnected)output.innerHTML=`<div class="form-status">${esc(error.message||'Retrieval is temporarily unavailable.')}</div>`;}
  finally{if(submit.isConnected)submit.disabled=false;}
 }
 if(event.target.id==='evidence-form'){event.preventDefault();state.query=$('#evidence-query').value.trim();state.section=$('#section-filter').value;state.form=$('#form-filter').value;$('#evidence-output').innerHTML=evidenceOutput();}
});
document.addEventListener('change',event=>{if(event.target.id==='as-of'&&/^\d{4}-\d{2}-\d{2}$/.test(event.target.value)&&event.target.value<=currentDate()){state.asOf=event.target.value;render();}if(['section-filter','form-filter'].includes(event.target.id))$('#evidence-form')?.requestSubmit();});
$('#company-search').addEventListener('input',event=>renderSearch(event.target.value));
$('#company-search').addEventListener('keydown',event=>{if(event.key!=='Enter')return;event.preventDefault();const symbol=event.target.value.trim().toUpperCase().replaceAll('.','-');const company=state.catalog.companies.find(c=>c.ticker.replaceAll('.','-')===symbol);if(company){$('#company-dialog').close();routeCompany(company.ticker);}else if(state.importsEnabled&&!state.importJob&&/^[A-Z][A-Z0-9-]{0,19}$/.test(symbol))startImport(symbol);});
$('#company-dialog').addEventListener('click',event=>{if(event.target===$('#company-dialog')){const rect=event.target.getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)event.target.close();}});
document.addEventListener('keydown',event=>{if(event.key==='/'&&!['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)){event.preventDefault();if(state.catalog)showSearch();}});
window.addEventListener('hashchange',route);


function riskOutput(result){
 const model=result.model||{};const inputs=result.inputs;const score=result.score;
 const title=score==null?'No released risk score':result.production_approval?'Released model output':'Experimental model output';
 const reasons=(result.reasons||[]).map(r=>`<p>${esc(r)}</p>`).join('');
 return `<div class="analysis-title-row"><h2>Bankruptcy disclosure model</h2><span class="review-status">${result.production_approval?'Released model':'Research only'}</span></div><p class="analysis-subtitle">${esc(model.target_definition||'Issuer Chapter 7/11 petition disclosed in SEC filings within one year.')}</p><div class="analysis-grid"><section class="panel"><div class="panel-header"><h3 class="panel-title">${title}</h3></div><div class="analysis-copy"><div class="model-score">${score==null?'Not released':(score*100).toFixed(2)+'%'}</div><div class="model-status">${score==null?'No risk category or alert is assigned.':esc(result.limitation||'Research model output')}</div>${reasons}<dl class="model-facts"><div><dt>Model</dt><dd>${esc(model.name||'Unavailable')}</dd></div><div><dt>Training cutoff</dt><dd>${date(model.trained_through)}</dd></div><div><dt>Decision threshold</dt><dd>${result.threshold==null?'Not released':esc(result.threshold)}</dd></div><div><dt>Feature source filed</dt><dd>${date(inputs?.feature_filing_date)}</dd></div></dl>${inputs?sourceLink(inputs.source_url,'Read the financial source filing ↗','research-link'):''}</div></section><section class="panel"><div class="panel-header"><h3 class="panel-title">Input coverage</h3></div><div class="analysis-copy"><div class="model-score">${inputs?inputs.observed_features+' / '+inputs.ordered_features.length:'Unavailable'}</div><p>Observed model inputs. Missing values remain unavailable; they are not treated as signs of financial health.</p><p>${inputs?.metadata_complete===false?'Filing-history coverage is incomplete. Event counts are withheld.':'Industry indicators may be missing when a dated SIC source is unavailable.'}</p></div></section></div>${!result.production_approval?'<div class="analysis-warning" style="margin-top:20px">Final review and independent evaluation remain incomplete. Research scores, when explicitly enabled on the server, are experimental and have no validated low/high-risk categories.</div>':''}${inputs?`<section class="panel" style="margin-top:22px"><div class="panel-header"><h3 class="panel-title">Model inputs</h3><span class="muted">${inputs.missing_features.length} unavailable</span></div><div class="table-wrap"><table><thead><tr><th>Feature</th><th class="numeric">Observed value</th><th>Status</th></tr></thead><tbody>${inputs.ordered_features.map(name=>`<tr><td class="feature-name">${esc(name.replaceAll('_',' '))}</td><td class="numeric">${inputs.values[name]==null?'—':esc(Number(inputs.values[name]).toLocaleString('en-US',{maximumFractionDigits:5}))}</td><td class="${inputs.values[name]==null?'missing-value':'muted'}">${inputs.values[name]==null?'Unavailable':'Observed'}</td></tr>`).join('')}</tbody></table></div></section>`:''}`;
}
function memoOutput(result){
 return `${combinedOutput(result.combined_assessment)}${synthesisOutput(result.synthesis)}<div class="analysis-title-row"><h2>Agent research evidence</h2><span class="badge">Cited excerpts</span></div><div class="analysis-warning">${esc(result.limitation)} These sections collect evidence for your review; they do not issue buy/sell calls.</div>${result.sections.map(section=>`<section class="research-section"><h3>${esc(section.title)}</h3><div class="evidence-results">${section.source_ids.map(id=>{const source=result.citations.find(c=>c.id===id);return `<div id="research-source-${id}">${evidenceCard({...source,filing_type:'SEC'},'',id-1)}</div>`;}).join('')||'<div class="empty-state">No verified source passages were available by this date.</div>'}</div></section>`).join('')}<div class="panel"><div class="panel-header"><h3 class="panel-title">Bankruptcy model</h3><button class="text-button" data-tab="risk">View inputs and status ↗</button></div><div class="analysis-copy">${esc(result.risk?.status||'Unavailable').replaceAll('_',' ')}. ${esc(result.risk?.reasons?.join(' ')||result.risk?.limitation||'')}</div></div>`;
}
function changesOutput(result){
 if(!result.reports.length)return '<div class="empty-state"><h3>Two comparable filings are needed.</h3>Import or refresh this company to obtain comparable original annual or quarterly reports.</div>';
 return `<div class="analysis-title-row"><h2>What changed in the filings?</h2><span class="badge">Source comparison</span></div><div class="analysis-warning">${esc(result.note)}</div>${result.reports.map(report=>`<section class="panel" style="margin-bottom:22px"><div class="panel-header"><h3 class="panel-title">Form ${esc(report.filing_type)} · ${date(report.previous.filing_date)} → ${date(report.current.filing_date)}</h3></div><div class="change-sources">${sourceLink(report.previous.source_url,'Read previous filing ↗')}${sourceLink(report.current.source_url,'Read current filing ↗')}</div>${report.sections.map(section=>`<div class="change-section"><h4>${esc(sectionName(section.section))}</h4>${section.status==='unavailable'?`<p class="analysis-subtitle">${esc(section.reason)}</p>`:`<p class="analysis-subtitle">${section.changes.length} text changes identified.</p><details><summary class="text-button">Read changes</summary>${section.changes.map(change=>`${change.current_text?`<div class="change-block"><small>Current filing · ${esc(change.kind)}</small><p>${esc(change.current_text)}</p></div>`:''}${change.previous_text?`<div class="change-block change-removed"><small>Previous filing</small><p>${esc(change.previous_text)}</p></div>`:''}`).join('')}</details>`}</div>`).join('')}</section>`).join('')}`;
}
async function loadAnalysis(){
 const output=$('#analysis-output');if(!output)return;
 if(!state.researchEnabled){output.innerHTML='<div class="empty-state"><h3>Connect the research server.</h3>Run the included Python server to use research memos, filing comparisons, and model-input checks.</div>';return;}
 const ticker=state.ticker,asOf=state.asOf,tab=state.tab;
 const endpoint={research:'research',risk:'risk',changes:'changes',market:'market'}[tab];
 output.innerHTML='<div class="loading"><span class="loader"></span>Checking dated sources and preparing analysis…</div>';
 try{
  const response=await fetch(`${state.apiBase}/api/${endpoint}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ticker,as_of:asOf})});
  if(response.status===401){output.innerHTML='<div class="empty-state"><h3>Sign in to use research tools.</h3><a class="button primary" href="/auth/login">Sign in / Join</a></div>';return;}if(!response.ok)throw Error('Analysis unavailable');const result=await response.json();
  if(!output.isConnected||state.ticker!==ticker||state.asOf!==asOf||state.tab!==tab)return;
  output.innerHTML=tab==='risk'?riskOutput(result):tab==='research'?memoOutput(result):tab==='market'?marketOutput(result):changesOutput(result);
 }catch{if(output.isConnected)output.innerHTML='<div class="empty-state"><h3>Verified analysis is unavailable.</h3>The server could not verify the required sources. Try a different date or refresh the company’s SEC data.<br><button class="button" data-action="retry-analysis">Try again</button></div>';}
}

function importProgress(job){
 if(job.status==='failed')return `<div class="import-box"><h3>Import not completed.</h3><p class="import-error">${esc(job.message)}</p><button class="button" data-action="dismiss-import">Try another ticker or retry</button></div>`;
 return `<div class="import-box"><h3>Importing ${esc(job.ticker)}</h3><div class="import-progress"><span class="loader"></span><span>${esc(job.stage)}${job.documents_total?` · ${job.documents_done||0}/${job.documents_total} reports`:''}</span></div><p class="import-warning">You can keep researching while this runs. Open company search again to see progress.</p></div>`;
}
async function startImport(ticker,refresh=false,history=false){
 if(!state.importsEnabled)return;
 state.importJob={ticker,status:'queued',stage:'Starting import'};renderSearch($('#company-search').value);
 try{
  const response=await fetch(`${state.apiBase}/api/imports`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ticker,refresh,history})});
  const job=await response.json();if(!response.ok)throw Error(job.error||'Could not start the import.');
  state.importJob=job;if(job.id)try{sessionStorage.setItem('ledger.import.job',job.id);}catch{}
  await pollImport();
 }catch(error){state.importJob={ticker,status:'failed',message:error.message};renderSearch($('#company-search').value);}
}
async function pollImport(){
 clearTimeout(importTimer);const job=state.importJob;if(!job)return;
 try{
  if(job.id){const response=await fetch(`${state.apiBase}/api/imports/${encodeURIComponent(job.id)}`);if(!response.ok)throw Error('Could not check the import. Open company search and retry.');state.importJob=await response.json();}
  if(state.importJob.status==='ready'){
   const ticker=state.importJob.ticker;const response=await fetch('./data/catalog.json',{cache:'no-store'});if(!response.ok)throw Error('Could not reload the company catalog.');state.catalog=await response.json();cache.delete(ticker);
   if(!state.stars.includes(ticker)){state.stars.push(ticker);store('ledger.watchlist.v1',state.stars);}
   state.asOf=state.importJob.as_of||state.asOf;state.importJob=null;try{sessionStorage.removeItem('ledger.import.job');}catch{}
   $('#company-dialog').close();toast(`${ticker} is ready to research`);routeCompany(ticker);await route();return;
  }
  if($('#company-dialog').open)renderSearch($('#company-search').value);
  if(state.importJob.status!=='failed')importTimer=setTimeout(pollImport,1500);
  else try{sessionStorage.removeItem('ledger.import.job');}catch{}
 }catch(error){state.importJob={ticker:job.ticker,status:'failed',message:error.message};if($('#company-dialog').open)renderSearch($('#company-search').value);}
}

async function init(){try{
 await window.ledgerAccountReady;
 if(window.ledgerAccount?.required){state.stars=[];state.notes={};}
 if(window.ledgerAccount?.user){
   const saved=await fetch('/api/me');
   if(!saved.ok)throw Error('Saved research unavailable');
   const data=await saved.json();state.stars=data.watchlist;state.notes=data.notes;
 }
const response=await fetch('./data/catalog.json');if(!response.ok)throw Error('Missing catalog');state.catalog=await response.json();const configResponse=await fetch('./config.json');if(configResponse.ok){const config=await configResponse.json();state.apiBase=typeof config.apiBase==='string'?config.apiBase:null;state.importsEnabled=config.imports===true;state.researchEnabled=config.research===true;if(state.apiBase!==null)state.asOf=currentDate();}$('#snapshot-label').textContent=date(state.catalog.snapshot_date);await route();if(state.importsEnabled){try{const saved=sessionStorage.getItem('ledger.import.job');if(saved){state.importJob={id:saved,ticker:'company',status:'running',stage:'Checking import progress'};pollImport();}}catch{}}}catch{main.innerHTML='<div class="empty-state error-state"><h3>The research snapshot could not load.</h3>Serve this folder over HTTP, rather than opening index.html directly.<code>python3 -m http.server 8000 --directory frontend</code>Then open localhost:8000.</div>';}}
init();

function synthesisOutput(result){if(!result||result.status!=='draft')return '';return `<section class="panel account-panel"><h3>AI research summary</h3><p class="analysis-subtitle">${esc(result.message)}</p>${result.statements.map(s=>`<p>${esc(s.text)} <strong>[${s.source_ids.map(esc).join(', ')}]</strong></p><details><summary>Supporting excerpt</summary><blockquote>${esc(s.quote)}</blockquote></details>`).join('')}</section>`;}
function marketOutput(result){if(result.status!=='available')return `<div class="empty-state"><h3>Market data unavailable</h3>${esc(result.message)}</div>`;const q=result.quote;return `<div class="analysis-title-row"><h2>Market & news</h2><span class="badge">${esc(result.provider)}</span></div><p class="analysis-subtitle">Current provider data, independent of the SEC evidence cutoff. ${esc(result.message)}</p><section class="panel account-panel"><h3>Latest provider quote</h3><div class="stat-value">${esc(q.c)}</div><p>Quote time: ${esc(new Date(q.t*1000).toLocaleString())}</p></section>${marketHistory(result.history||[])}<section class="panel account-panel"><h3>Provider valuation ratios</h3>${Object.entries(result.metrics||{}).map(([k,v])=>`<p>${esc(({peTTM:'P/E · trailing 12 months',pb:'Price / book',psTTM:'Price / sales · TTM',marketCapitalization:'Market capitalization · USD millions',dividendYieldIndicatedAnnual:'Indicated dividend yield · %'})[k]||k)}: <strong>${esc(Number(v).toLocaleString('en-US',{maximumFractionDigits:2}))}</strong></p>`).join('')||'<p>Valuation data unavailable for this symbol or plan.</p>'}</section>${result.news.map(n=>`<article class="panel account-panel"><h3><a href="${esc(n.url)}" target="_blank" rel="noopener noreferrer">${esc(n.headline)}</a></h3><p>${esc(n.source)} · ${esc(new Date(n.datetime*1000).toLocaleDateString())}</p></article>`).join('')}`;}

function marketHistory(points){if(points.length<2)return '<section class="panel account-panel"><h3>Price history</h3><p>Historical prices unavailable for this provider plan or symbol.</p></section>';const w=700,h=220,pad=25;const prices=points.map(p=>p.close),lo=Math.min(...prices),hi=Math.max(...prices),span=Math.max(hi-lo,1);const coords=points.map((p,i)=>`${pad+i*(w-2*pad)/(points.length-1)},${h-pad-(p.close-lo)/span*(h-2*pad)}`).join(' ');return `<section class="panel account-panel"><h3>Daily closing price · provider history</h3><svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Daily provider closing price"><polyline points="${coords}" fill="none" stroke="#2677cb" stroke-width="2"/></svg><p>${esc(new Date(points[0].timestamp*1000).toLocaleDateString())} – ${esc(new Date(points.at(-1).timestamp*1000).toLocaleDateString())}. Historical adjustment conventions depend on provider.</p></section>`;}

function combinedOutput(assessment){if(!assessment)return '';const m=assessment.model;return `<div class="analysis-title-row"><h2>Model + research agents</h2><span class="badge">${assessment.status==='screening_available'?'Screening available':'Evidence review only'}</span></div><section class="panel account-panel"><h3>Bankruptcy model signal</h3><div class="stat-value">${m.score==null?'Score withheld':esc((m.score*100).toFixed(2))+'%'}</div><p>${esc(m.target)}</p><p>${m.screening_signal==='unavailable'?'No released screening signal.':esc(m.screening_signal.replaceAll('_',' '))}</p><p>Financial source: ${date(m.feature_filing_date)} · Observed inputs: ${m.observed_inputs??'Unavailable'} · Missing: ${m.missing_inputs??'Unavailable'}</p>${m.reasons.map(r=>`<p>${esc(r)}</p>`).join('')}<button class="text-button" data-tab="risk">Inspect model inputs and release status ↗</button></section><p class="analysis-subtitle">${esc(assessment.explanation)}</p><section class="panel"><div class="panel-header"><h3 class="panel-title">Financial inputs alongside filing evidence</h3></div><div class="table-wrap"><table><thead><tr><th>Research topic</th><th>Observed financial inputs</th><th>Related passages</th></tr></thead><tbody>${assessment.themes.map(t=>`<tr><td>${esc(t.title)}</td><td class="feature-name">${t.inputs.map(v=>`${esc(v.feature.replaceAll('_',' '))}: <strong>${v.value==null?'Unavailable':esc(Number(v.value).toLocaleString('en-US',{maximumFractionDigits:4}))}</strong>`).join('<br>')}</td><td>${t.source_ids.map(id=>`<button class="text-button" data-source-id="${id}">[${id}]</button>`).join(' ')||'No matching passage'}</td></tr>`).join('')}</tbody></table></div></section><section class="panel account-panel"><h3>What still needs review</h3><ul>${[...assessment.gaps,...assessment.next_steps].map(s=>`<li>${esc(s)}</li>`).join('')}</ul></section>`;}
