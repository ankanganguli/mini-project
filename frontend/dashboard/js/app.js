const state = {
  mode: "text",
  currentResult: null,
  history: [
    {id:1,name:"News claim sample",type:"Text",score:92.3,status:"High confidence",time:"Today, 2:14 PM"},
    {id:2,name:"portrait_sample.jpg",type:"Image",score:18.6,status:"Needs review",time:"Today, 11:40 AM"},
    {id:3,name:"video_sample.mp4",type:"Video",score:76.8,status:"High confidence",time:"Yesterday"},
    {id:4,name:"Social media post",type:"Text",score:64.2,status:"Needs review",time:"Yesterday"},
    {id:5,name:"sample-image.webp",type:"Image",score:88.1,status:"High confidence",time:"Aug 21"}
  ]
};

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

function showPage(page) {
  $$(".page").forEach(p => p.classList.toggle("active", p.id === page));
  $$(".nav-item").forEach(n => n.classList.toggle("active", n.dataset.page === page));
  const titles = {
    dashboard:"Good afternoon 👋", analyze:"New analysis", result:"Analysis result",
    history:"Analysis history", reports:"Reports", settings:"Settings"
  };
  $("#pageTitle").textContent = titles[page] || "VeritiScan AI";
  window.scrollTo({top:0,behavior:"smooth"});
}

document.addEventListener("click", e => {
  const target = e.target.closest("[data-page]");
  if (target) showPage(target.dataset.page);
});

const logoutBtn = document.getElementById("logoutBtn");
if (logoutBtn) {
  logoutBtn.addEventListener("click", () => {
    localStorage.removeItem("isLoggedIn");
    localStorage.removeItem("userIdentifier");
    window.location.href = "../login.html";
  });
}

function renderRecent() {
  $("#recentList").innerHTML = state.history.slice(0,4).map(item => `
    <div class="analysis-row">
      <div class="type-icon ${item.type.toLowerCase()}">${item.type === "Text" ? "Aa" : item.type === "Image" ? "◉" : "▶"}</div>
      <div><div class="row-title">${item.name}</div><div class="row-sub">${item.type} • ${item.time}</div></div>
      <div class="score">${item.score.toFixed(1)}%</div>
      <div class="status ${item.status.includes("review") ? "review":""}">${item.status}</div>
    </div>`).join("");
}

function renderHistory() {
  const q = ($("#historySearch")?.value || "").toLowerCase();
  const filter = $("#historyFilter")?.value || "all";
  const data = state.history.filter(x => (filter === "all" || x.type === filter) && x.name.toLowerCase().includes(q));
  $("#historyList").innerHTML = `
    <div class="history-item header"><span>Analysis</span><span>Type</span><span>Score</span><span>Status</span></div>
    ${data.map(item => `
      <div class="history-item">
        <div class="history-main">${item.name}<small>${item.time}</small></div>
        <div>${item.type}</div>
        <div><strong>${item.score.toFixed(1)}%</strong></div>
        <div class="status ${item.status.includes("review") ? "review":""}">${item.status}</div>
      </div>`).join("") || `<div class="empty-state"><h3>No analyses found</h3><p>Try a different search or filter.</p></div>`}`;
}

function setMode(mode) {
  state.mode = mode;
  $$(".mode-card").forEach(c => c.classList.toggle("selected", c.dataset.mode === mode));
  const configs = {
    text:{title:"Text to analyze",hint:"Paste the content you want VeritiScan AI to examine.",label:"Analyze text",type:"TEXT"},
    image:{title:"Image to analyze",hint:"Upload an image for the model to analyze.",label:"Analyze image",type:"IMAGE"},
    video:{title:"Video to analyze",hint:"Upload a video for the model to analyze.",label:"Analyze video",type:"VIDEO"}
  };
  const c = configs[mode];
  $("#inputTitle").textContent=c.title; $("#inputHint").textContent=c.hint; $("#analyzeBtn").innerHTML=`${c.label} <span>✦</span>`; $("#inputType").textContent=c.type;
  $("#inputArea").classList.toggle("hidden",mode!=="text");
  $("#fileArea").classList.toggle("hidden",mode==="text");
  if(mode!=="text"){ $("#fileInput").accept=mode==="image"?"image/*":"video/*"; $("#dropTitle").textContent=`Drop your ${mode} here`; $("#fileMeta").textContent=mode==="image"?"PNG, JPG, WEBP • Prototype limit 20 MB":"MP4, MOV, WEBM • Prototype limit 100 MB"; $("#preview").classList.add("hidden"); $("#preview").innerHTML=""; $("#fileInput").value=""; }
}

$$(".mode-card").forEach(c => c.addEventListener("click",()=>setMode(c.dataset.mode)));
$("#textInput").addEventListener("input", e => $("#charCount").textContent=`${e.target.value.length} characters`);

$("#fileInput").addEventListener("change", e => {
  const file=e.target.files[0]; if(!file) return;
  const preview=$("#preview"); preview.classList.remove("hidden");
  const url=URL.createObjectURL(file);
  preview.innerHTML = state.mode==="image" ? `<img src="${url}" alt="Selected image preview">` : `<video src="${url}" controls></video>`;
  $("#dropTitle").textContent=file.name;
  $("#fileMeta").textContent=`${(file.size/1024/1024).toFixed(2)} MB • Ready for demo analysis`;
});

function fakeAnalysis() {
  const scores={text:87.4,image:74.2,video:81.6};
  const score=scores[state.mode]+(Math.random()*4-2);
  const rounded=Math.max(1,Math.min(99.9,score));
  let name=state.mode==="text" ? ($("#textInput").value.trim().slice(0,34) || "Text submission") : ($("#fileInput").files[0]?.name || `${state.mode} submission`);
  return {score:rounded,name,type:state.mode[0].toUpperCase()+state.mode.slice(1)};
}

$("#analyzeBtn").addEventListener("click",()=>{
  if(state.mode==="text" && !$("#textInput").value.trim()){ showToast("Add some text before analyzing."); return; }
  if(state.mode!=="text" && !$("#fileInput").files[0]){ showToast(`Choose a ${state.mode} first.`); return; }
  const btn=$("#analyzeBtn"); const old=btn.innerHTML; btn.disabled=true; btn.innerHTML="Analyzing <span>◌</span>";
  setTimeout(()=>{
    const r=fakeAnalysis(); state.currentResult=r;
    state.history.unshift({id:Date.now(),name:r.name,type:r.type,score:r.score,status:r.score>=75?"High confidence":"Needs review",time:"Just now"});
    populateResult(r); renderRecent(); renderHistory(); showPage("result"); btn.disabled=false; btn.innerHTML=old; showToast("Analysis complete");
  },1100);
});

function populateResult(r){
  const positive=r.score>=75;
  $("#confidenceValue").textContent=r.score.toFixed(1)+"%";
  $("#detailConfidence").textContent=r.score.toFixed(1)+"%";
  $("#resultLabel").textContent=positive?"Likely authentic":"Needs further review";
  $("#resultDescription").textContent=positive?"The prototype model returned a relatively high-confidence result for this submission.":"The prototype model returned a lower-confidence result. Review the details before drawing conclusions.";
  $("#resultType").textContent=r.type.toUpperCase(); $("#detailType").textContent=r.type;
  $("#confidenceRing").style.background=`conic-gradient(#5f70f3 0 ${r.score}%,#e9ebf3 ${r.score}% 100%)`;
  if(r.type==="Text") $("#submissionPreview").textContent=$("#textInput").value.trim();
  else $("#submissionPreview").innerHTML=`<strong>${r.name}</strong><br><br>Preview was selected during this demo session.`;
}

$("#historySearch").addEventListener("input",renderHistory);
$("#historyFilter").addEventListener("change",renderHistory);

function showToast(message){const t=$("#toast");t.textContent=message;t.classList.add("show");clearTimeout(window.__toast);window.__toast=setTimeout(()=>t.classList.remove("show"),2200)}

renderRecent(); renderHistory(); setMode("text");
