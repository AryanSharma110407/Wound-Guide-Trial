import React, {useState} from 'react'
import {createRoot} from 'react-dom/client'
import './style.css'

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

function App(){
  const [file,setFile]=useState(null), [items,setItems]=useState(''), [result,setResult]=useState(null), [busy,setBusy]=useState(false), [error,setError]=useState('')
  async function analyze(){
    if(!file){setError('Please select a wound image first.');return}
    setBusy(true); setError(''); setResult(null)
    const form=new FormData(); form.append('image',file); form.append('items',items)
    try{
      const r=await fetch(`${API}/api/analyze`,{method:'POST',body:form})
      const data=await r.json(); if(!r.ok) throw new Error(data.detail||'Analysis failed')
      setResult(data)
    }catch(e){setError(e.message)}finally{setBusy(false)}
  }
  return <main className="page">
    <header><div><span className="eyebrow">AI FIRST-AID PROTOTYPE</span><h1>WoundGuide</h1><p>Classify a cut, burn, or bruise and get evidence-based temporary first-aid guidance based on the items you have.</p></div><div className="status">{busy?'Analyzing…':'Ready'}</div></header>
    <section className="grid">
      <div className="card input-card">
        <h2>1. Show the injury</h2>
        <label className="drop"><input type="file" accept="image/*" capture="environment" onChange={e=>setFile(e.target.files?.[0]||null)}/><strong>{file?file.name:'Choose or capture an image'}</strong><span>JPG, PNG or WEBP · up to 10 MB</span></label>
        <h2>2. What do you have?</h2>
        <textarea value={items} onChange={e=>setItems(e.target.value)} placeholder="Example: clean water, clean cloth, gauze, ice pack" />
        <button onClick={analyze} disabled={busy}>{busy?'Analyzing…':'Analyze & Get Guidance'}</button>
        {error&&<div className="error">{error}</div>}
      </div>
      <div className="card result-card">
        {!result?<div className="empty"><div className="icon">+</div><h2>Your guidance will appear here</h2><p>The ML model identifies the injury category first. The knowledge base then supplies the facts used by the selected LLM provider.</p></div>:<>
          <div className="result-head"><div><span className="eyebrow">MODEL RESULT</span><h2>{result.prediction.injury}</h2></div><div className="confidence">{result.prediction.confidence}%<small>confidence</small></div></div>
          <div className="bar"><span style={{width:`${result.prediction.confidence}%`}}/></div>
          <div className="guidance">{result.guidance.split('\n').map((x,i)=><p key={i}>{x}</p>)}</div>
          {result.matched_materials.length>0&&<div><h3>Matched items</h3><div className="chips">{result.matched_materials.map((m,i)=><span key={i}>{m.item}</span>)}</div></div>}
          <div className="notice">This is a college-project prototype for first-aid education, not a medical diagnosis. Seek professional care for serious, worsening, or uncertain injuries.</div>
        </>}
      </div>
    </section>
    <footer>Provider: <strong>{result?.provider||'configured on backend'}</strong> · No automatic provider fallback</footer>
  </main>
}
createRoot(document.getElementById('root')).render(<App/>)
