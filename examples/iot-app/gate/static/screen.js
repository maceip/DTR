const select = document.getElementById('scenario'), terminal = document.getElementById('terminal')
fetch('/api/status').then(r => r.json()).then(data => {
  const update = () => terminal.textContent = `proposed command:\n\n$ ${data.scenarios[+select.value]}\n\nrun this command? [y/N]`
  select.onchange = update; update()
})
