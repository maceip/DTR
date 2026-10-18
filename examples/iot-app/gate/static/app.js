const $ = id => document.getElementById(id)
let scenarios = [], busy = false, lastResult = null
function show(result) {
  lastResult = result.id
  $('checked-command').textContent = '$ ' + result.command
  $('verdict').textContent = result.verdict
  $('verdict').className = `verdict ${result.verdict}`
  $('reason').textContent = result.reason
  $('frame-source').textContent = result.source === 'camera' ? 'live brio camera' : 'rendered terminal'
  $('led').textContent = result.indicator?.status === 'complete' ? (result.indicator.result.enabled ? 'warning on · read back' : 'clear · read back') : 'unavailable'
  $('tool').textContent = result.tool?.status === 'complete' ? `tests exited ${result.tool.result.exit_code}` : result.tool?.status || 'not run'
  $('elapsed').textContent = `${(result.elapsed_ms / 1000).toFixed(1)}s`
  for (const [model, element] of [['screen', 'vision-runtime'], ['judge', 'judge-runtime']]) {
    const execution = result.models?.[model]?.execution
    $(element).textContent = execution ? `${execution.backend} · ${execution.requested_accelerator}${execution.npu_execution_verified ? ' ✓' : ''}` : 'not run'
  }
  $('raw').textContent = JSON.stringify(result, null, 2)
  $('frame').src = `/api/frame?t=${Date.now()}`
  $('frame').hidden = false
}
async function status() {
  try {
    const response = await fetch('/api/status'), data = await response.json()
    if (!scenarios.length) {
      scenarios = data.scenarios
      $('intent').value = data.intent
      $('command').textContent = '$ ' + scenarios[+$('scenario').value]
      if (data.last_result) show(data.last_result)
    }
    $('connection').textContent = data.progress.stage === 'ready' ? 'runtime connected' : data.progress.stage
    if (data.camera) {
      const c = data.camera
      $('video-status').textContent = c.error ? c.error.message : `${c.frames} frames · ${c.fps} fps${c.covered ? ' · image is black — check the shutter' : ''}`
    }
    if (!busy && data.last_result && (data.last_result.id !== lastResult || $('verdict').textContent === 'checking')) show(data.last_result)
    if (!busy && data.progress.stage !== 'ready') {
      $('verdict').textContent = 'checking'; $('verdict').className = 'verdict ready'
      $('reason').textContent = 'the board is reading the image and comparing it with your intent'
    }
    $('check').disabled = busy || data.progress.stage !== 'ready'
    const stages = ['reading screen', 'checking intent', 'applying permissions']
    stages.forEach((name, i) => $('s' + (i + 1)).className = data.progress.stage === name ? 'active' : '')
  } catch { $('connection').textContent = 'runtime disconnected' }
}
$('scenario').onchange = () => $('command').textContent = '$ ' + scenarios[+$('scenario').value]
$('source').onchange = () => {
  const live = $('source').value === 'camera'
  $('video-panel').hidden = !live
  if (live) $('live-video').src = '/api/video.mjpg'
  else $('live-video').removeAttribute('src')
}
if (new URLSearchParams(location.search).get('source') === 'camera') {
  $('source').value = 'camera'; $('source').onchange()
}
$('check').onclick = async () => {
  if (busy) return
  busy = true; $('check').disabled = true; $('check').textContent = 'checking locally…'
  $('verdict').textContent = 'checking'; $('verdict').className = 'verdict ready'
  $('reason').textContent = 'the board is reading the image and comparing it with your intent'
  try {
    const response = await fetch('/api/check', {method:'POST', headers:{'Content-Type':'application/json','X-Gate-Request':'1'}, body:JSON.stringify({command:scenarios[+$('scenario').value], intent:$('intent').value, source:$('source').value, execute:true})})
    const result = await response.json()
    if (!response.ok) throw new Error(result.error?.message || JSON.stringify(result))
    show(result)
  } catch (error) { $('verdict').textContent = 'review'; $('verdict').className = 'verdict review'; $('reason').textContent = error.message }
  finally { busy = false; $('check').disabled = false; $('check').textContent = 'check the command →'; status() }
}
status(); setInterval(status, 1500)
