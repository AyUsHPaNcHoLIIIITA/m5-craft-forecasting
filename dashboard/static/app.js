document.addEventListener('DOMContentLoaded', () => {
    const itemSel   = document.getElementById('item_id');
    const storeSel  = document.getElementById('store_id');
    const dateSel   = document.getElementById('start_date');
    const hrzInput  = document.getElementById('horizon');
    const hrzVal    = document.getElementById('horizon-val');
    const form      = document.getElementById('forecast-form');
    const actualsOn = document.getElementById('show-actuals');

    let mainChart = null, routeChart = null;
    let allDates = [];              // [{date, horizon, wday, event}, ...]

    // ── Slider label ──
    hrzInput.addEventListener('input', e => {
        hrzVal.textContent = e.target.value;
        // Dynamically cap max based on start date
        updateMaxHorizon();
    });

    // ── Load options ──
    fetch('/api/options')
        .then(r => r.json())
        .then(data => {
            // Items
            itemSel.innerHTML = '';
            data.items.forEach(it => {
                const o = document.createElement('option');
                o.value = it; o.textContent = it;
                itemSel.appendChild(o);
            });
            // Stores
            storeSel.innerHTML = '';
            data.stores.forEach(st => {
                const o = document.createElement('option');
                o.value = st; o.textContent = st;
                storeSel.appendChild(o);
            });
            // Dates
            allDates = data.dates;
            dateSel.innerHTML = '';
            data.dates.forEach(d => {
                const o = document.createElement('option');
                o.value = d.horizon;
                let label = d.date;
                if (d.wday == 1 || d.wday == 2) label += '  ⟵ Weekend';
                if (d.event) label += `  🎉 ${d.event}`;
                o.textContent = label;
                dateSel.appendChild(o);
            });

            updateMaxHorizon();
            generateForecast();
        })
        .catch(e => console.error('Options error:', e));

    // When start date changes, update max horizon
    dateSel.addEventListener('change', updateMaxHorizon);

    function updateMaxHorizon() {
        const startH = parseInt(dateSel.value) || 1;
        const maxH = 7 - startH + 1; // can't go past day 7
        hrzInput.max = maxH;
        if (parseInt(hrzInput.value) > maxH) {
            hrzInput.value = maxH;
            hrzVal.textContent = maxH;
        }
    }

    // ── Submit ──
    form.addEventListener('submit', e => { e.preventDefault(); generateForecast(); });
    actualsOn.addEventListener('change', () => {
        if (window._lastData) renderMain(window._lastData);
    });

    function generateForecast() {
        const item    = itemSel.value;
        const store   = storeSel.value;
        const startH  = dateSel.value;
        const horizon = hrzInput.value;
        if (!item || !store || !startH) return;

        const btn = document.getElementById('btn-generate');
        btn.textContent = 'Processing…'; btn.style.opacity = '0.6';

        fetch(`/api/forecast?item_id=${item}&store_id=${store}&start_horizon=${startH}&horizon=${horizon}`)
            .then(r => r.json())
            .then(data => {
                if (data.error) { alert(data.error); return; }
                window._lastData = data;

                // KPIs
                document.getElementById('kpi-total').textContent = data.kpi.total_expected;
                document.getElementById('kpi-avg').textContent   = data.kpi.avg_daily;

                // Context card
                const ctx = data.context;
                const dayClass = ctx.is_weekend ? 'ctx-weekend' : 'ctx-weekday';
                const dayLabel = ctx.is_weekend ? '🗓 Weekend' : '🗓 Weekday';
                const regClass = ctx.regime === 'Volatile' ? 'ctx-volatile' : 'ctx-stable';
                const eventHtml = ctx.event !== 'None'
                    ? `<span class="ctx-event">🎉 ${ctx.event}</span>`
                    : '<span class="ctx-weekday">None</span>';

                document.getElementById('context-info').innerHTML = `
                    <span class="ctx-label">Date:</span> <span class="ctx-val">${ctx.date}</span>
                    &nbsp;·&nbsp;
                    <span class="${dayClass}">${dayLabel}</span>
                    &nbsp;·&nbsp;
                    <span class="ctx-label">Event:</span> ${eventHtml}
                    <br>
                    <span class="ctx-label">Volatility:</span>
                    <span class="${regClass}">${ctx.volatility} (${ctx.regime})</span>
                    &nbsp;·&nbsp;
                    <span class="ctx-label">Gate →</span>
                    <span style="color:var(--primary-color)">${(ctx.tft_weight*100).toFixed(0)}% TFT</span>
                    /
                    <span style="color:var(--secondary-color)">${(ctx.lgb_weight*100).toFixed(0)}% LGB</span>
                `;

                renderMain(data);
                renderRouting(data);
            })
            .catch(e => { console.error(e); alert('Error fetching forecast'); })
            .finally(() => { btn.textContent = 'Generate Forecast'; btn.style.opacity = '1'; });
    }

    // ── Main Chart ──────────────────────────
    function renderMain(data) {
        const ctx = document.getElementById('forecastChart').getContext('2d');
        if (mainChart) mainChart.destroy();

        const showAct = actualsOn.checked;
        const labels  = [...data.history.labels, ...data.forecast.labels];

        // Build data arrays with nulls for gaps
        const nH = data.history.labels.length;
        const nF = data.forecast.labels.length;

        const histData = [...data.history.values, ...Array(nF).fill(null)];
        const craftData = [...Array(nH).fill(null), ...data.forecast.craft];
        const actualData = [...Array(nH).fill(null), ...data.forecast.actual];

        // Connect history → forecast seamlessly
        if (nH > 0) {
            const bridge = data.history.values[nH - 1];
            craftData[nH - 1]  = bridge;
            actualData[nH - 1] = bridge;
        }

        const datasets = [
            {
                label: 'History',
                data: histData,
                borderColor: 'rgba(255,255,255,0.35)',
                backgroundColor: 'rgba(255,255,255,0.05)',
                borderWidth: 2, tension: 0.3, pointRadius: 0,
            },
            {
                label: 'CRAFT Forecast',
                data: craftData,
                borderColor: '#00f0ff',
                backgroundColor: 'rgba(0,240,255,0.08)',
                borderWidth: 3, tension: 0.3,
                pointBackgroundColor: '#00f0ff', pointRadius: 4, pointHoverRadius: 7,
                fill: true,
            },
        ];
        if (showAct) {
            datasets.push({
                label: 'Actual Sales',
                data: actualData,
                borderColor: 'rgba(255,0,60,0.6)',
                borderWidth: 2, borderDash: [5,5], tension: 0.3, pointRadius: 0,
            });
        }

        mainChart = new Chart(ctx, {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true, maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { labels: { color: '#8b92a5', font: { size: 11 } } },
                    tooltip: {
                        backgroundColor: 'rgba(0,0,0,0.85)',
                        borderColor: 'rgba(0,240,255,0.25)', borderWidth: 1,
                    },
                },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#8b92a5', maxTicksLimit: 12 } },
                    y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#8b92a5' }, beginAtZero: true },
                },
            },
        });
    }

    // ── Routing Chart ───────────────────────
    function renderRouting(data) {
        const ctx = document.getElementById('routingChart').getContext('2d');
        if (routeChart) routeChart.destroy();

        routeChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.routing.labels,
                datasets: [
                    {
                        label: 'TFT Weight',
                        data: data.routing.w_tft,
                        backgroundColor: 'rgba(0,240,255,0.75)',
                        barPercentage: 0.9, categoryPercentage: 0.9,
                    },
                    {
                        label: 'LGB Weight',
                        data: data.routing.w_lgb,
                        backgroundColor: 'rgba(255,0,60,0.75)',
                        barPercentage: 0.9, categoryPercentage: 0.9,
                    },
                ],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    x: { stacked: true, ticks: { color: '#8b92a5', font: { size: 10 } } },
                    y: {
                        stacked: true, min: 0, max: 1,
                        ticks: { color: '#8b92a5', stepSize: 0.25,
                            callback: v => (v * 100) + '%' },
                        grid: { color: 'rgba(255,255,255,0.04)' },
                    },
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: c => c.dataset.label + ': ' + (c.raw * 100).toFixed(1) + '%',
                        },
                    },
                },
            },
        });
    }
});
