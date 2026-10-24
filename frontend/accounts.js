'use strict';
window.ledgerAccount = {required:false, user:null};
window.ledgerAccountReady = (async () => {
  const response = await fetch('/api/session');
  if (!response.ok || !(response.headers.get('content-type')||'').includes('application/json')) return;
  Object.assign(window.ledgerAccount, await response.json());
  if (!window.ledgerAccount.required) return;
  const button = document.createElement('button');
  button.className='tab';
  button.textContent=window.ledgerAccount.user ? 'Sign out' : 'Sign in / Join';
  button.onclick=async () => {
    if(window.ledgerAccount.user) await fetch('/api/logout',{method:'POST',redirect:'manual'});
    location.href=window.ledgerAccount.user ? window.ledgerAccount.logout_url : '/auth/login';
  };
  document.querySelector('.topbar').append(button);
})();

window.renderAccount = async function(view){
 const root=document.querySelector('#main');
 const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 if(!window.ledgerAccount.user){root.innerHTML='<h1>Account</h1><p>Sign in to manage your research and filing alerts.</p><a class="button primary" href="/auth/login">Sign in / Join</a>';return;}
 root.innerHTML='<div class="loading">Loading your account…</div>';
 try{
 const endpoints=view==='notifications'?['/api/notifications']:['/api/account/preferences','/api/account/usage','/api/capabilities'];
 const results=await Promise.all(endpoints.map(async url=>{const r=await fetch(url);if(!r.ok)throw Error('Unable to load account');return r.json();}));
 if(view==='notifications'){
 const alerts=results[0].notifications;
 root.innerHTML='<h1>Filing alerts</h1><p class="analysis-subtitle">Hourly SEC checks for your followed companies. Enable alerts in Account settings. First check establishes a baseline.</p><button class="button" id="read-alerts">Mark all read</button>'+alerts.map(a=>`<article class="panel account-panel"><strong>${escape(a.ticker)} · Form ${escape(a.filing_type)}</strong><p>${escape(a.filing_date)} ${a.read?'':' · New'}</p><a class="text-button" href="${/^https:\/\/www\.sec\.gov\//.test(a.source_url)?escape(a.source_url):'#'}" target="_blank" rel="noopener noreferrer">Read filing ↗</a></article>`).join('')+(!alerts.length?'<div class="empty-state">No new filing alerts.</div>':'');
 document.querySelector('#read-alerts').onclick=async()=>{await fetch('/api/notifications',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});window.renderAccount('notifications');};return;
 }
 const [prefs,usage,caps]=results;
 root.innerHTML=`<h1>Account settings</h1><section class="panel account-panel"><h2>Your account</h2><p>${escape(window.ledgerAccount.user.email)}</p><p>Saved research is private to your account.</p><button class="button" id="export-account">Export my data</button></section><section class="panel account-panel"><h2>Filing alerts</h2><label><input type="checkbox" id="filing-alerts" ${prefs.filing_alerts?'checked':''}> Notify me in Ledger about new filings from followed companies</label><p>Checked hourly. Public SEC reports are shared; personal notes are private. Email delivery is not configured.</p><div id="pref-status" role="status"></div></section><section class="panel account-panel"><h2>Usage</h2>${usage.usage.map(u=>`<p>${escape(({requests:'Requests this minute',daily:'Research requests · UTC day',imports:'Imports · UTC day'})[u.name])}: <strong>${u.used} / ${u.limit}</strong></p>`).join('')}</section><section class="panel account-panel"><h2>Connected services</h2><p>Filing retrieval: keyword + local latent semantic search</p><p>AI summaries: ${caps.ai_synthesis?'Connected':'Provider not configured'}</p><p>Market prices and news: ${caps.market_data?'Connected':'Provider not configured'}</p></section><section class="panel account-panel"><h2>Delete application account</h2><p>Deletes saved notes, watchlists, notifications and sessions, and blocks this identity from signing back into Ledger. Shared public filings and existing backups remain. ${caps.full_identity_deletion?'The Cognito login identity will also be deleted.':'Cognito login identity removal requires the site owner.'}</p><label>Type DELETE to confirm <input id="delete-confirm" autocomplete="off"></label><button class="button" id="delete-account">Delete application account</button><p id="delete-status" role="status"></p></section>`;
 document.querySelector('#filing-alerts').onchange=async e=>{const r=await fetch('/api/account/preferences',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filing_alerts:e.target.checked})});document.querySelector('#pref-status').textContent=r.ok?'Preferences saved.':'Could not save. Retry.';};
 document.querySelector('#export-account').onclick=async()=>{const r=await fetch('/api/account/export');if(!r.ok)return;const blob=new Blob([JSON.stringify(await r.json(),null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='ledger-account-export.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
 document.querySelector('#delete-account').onclick=async()=>{if(document.querySelector('#delete-confirm').value!=='DELETE')return;const r=await fetch('/api/account/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confirmation:'DELETE'})});if(r.ok)location.href='/';else document.querySelector('#delete-status').textContent='Deletion failed. Please retry.';};
 }catch{root.innerHTML='<div class="empty-state">Account settings are unavailable. Please sign in again or retry.</div>';}
};
