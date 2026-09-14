/**
 * SMVMD Interactive Pediatric EEG Diagnostic Client
 */

let currentSample = null;
let energyChartInstance = null;

document.addEventListener("DOMContentLoaded", () => {
    // Sliders & Controls
    const alphaSlider = document.getElementById("alphaSlider");
    const alphaVal = document.getElementById("alphaVal");
    alphaSlider.addEventListener("input", (e) => {
        alphaVal.textContent = e.target.value;
    });

    document.getElementById("btnLoadSample").addEventListener("click", loadSampleCohort);
    document.getElementById("btnRunDecomposition").addEventListener("click", runSMVMDAndDiagnose);

    // Initial load
    loadSampleCohort();
});

async function loadSampleCohort() {
    const disorder = document.getElementById("cohortSelect").value;
    const duration = document.getElementById("durationSelect").value;
    const btn = document.getElementById("btnLoadSample");

    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Loading EEG...`;

    try {
        const res = await fetch(`/api/generate_sample?disorder=${disorder}&duration=${duration}&fs=128`);
        const data = await res.json();
        currentSample = data;

        // Render Raw Multi-channel EEG Canvas
        drawMultiChannelEEG(data);

        // Reset Modes and Diagnostic Results
        document.getElementById("modesContainer").innerHTML = `
            <div class="placeholder-state">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
                <p>Click <strong>"Run SMVMD & Diagnose"</strong> to decompose the multi-channel EEG into its joint compact variational modes.</p>
            </div>
        `;
        document.getElementById("modesExtractedBadge").textContent = "Awaiting Decomposition";
        document.getElementById("channelCountBadge").textContent = `${data.channels.length} Channels @ ${data.fs} Hz`;

        resetDiagnosticBadges();
    } catch (err) {
        console.error("Failed to load sample EEG:", err);
        alert("Error loading EEG sample. Make sure backend is running.");
    } finally {
        btn.disabled = false;
        btn.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
            Generate Cohort EEG
        `;
    }
}

function drawMultiChannelEEG(sampleData) {
    const canvas = document.getElementById("rawEegCanvas");
    const ctx = canvas.getContext("2d");

    // Dynamic width & height resolution
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = 240 * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const W = rect.width;
    const H = 240;

    ctx.clearRect(0, 0, W, H);

    const channels = sampleData.channels;
    const C = channels.length;
    const T = channels[0].values.length;
    const spacing = H / (C + 1);

    // Color theme
    ctx.lineWidth = 1.2;

    for (let c = 0; c < C; c++) {
        const yOffset = (c + 1) * spacing;
        const vals = channels[c].values;
        const name = channels[c].name;

        // Channel Label
        ctx.fillStyle = "#9ca3af";
        ctx.font = "bold 10px JetBrains Mono";
        ctx.fillText(name, 10, yOffset - 4);

        // Base guideline
        ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
        ctx.beginPath();
        ctx.moveTo(40, yOffset);
        ctx.lineTo(W - 10, yOffset);
        ctx.stroke();

        // Signal trace
        ctx.strokeStyle = c < 4 ? "#00d4ff" : "#818cf8";
        ctx.beginPath();

        const xStep = (W - 55) / (T - 1);
        for (let t = 0; t < T; t++) {
            const x = 45 + t * xStep;
            const y = yOffset - vals[t] * 4.5;
            if (t === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }
}

async function runSMVMDAndDiagnose() {
    if (!currentSample) {
        alert("Please load an EEG sample first!");
        return;
    }

    const btn = document.getElementById("btnRunDecomposition");
    const alpha = parseFloat(document.getElementById("alphaSlider").value);

    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Decomposing via SMVMD...`;

    try {
        const res = await fetch("/api/decompose_and_diagnose", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                channels: currentSample.channels,
                fs: currentSample.fs,
                alpha: alpha,
                max_modes: 6,
            }),
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Decomposition failed");

        // 1. Render Extracted MVMFs
        renderDecomposedModes(data.decomposition);

        // 2. Render Real-Time Diagnostics
        renderDiagnosticResults(data.prediction, data.biomarkers);

        // 3. Render Energy Distribution Chart
        renderEnergyChart(data.decomposition.modes);

    } catch (err) {
        console.error("Diagnosis error:", err);
        alert(`Analysis Error: ${err.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            Run SMVMD & Diagnose
        `;
    }
}

function renderDecomposedModes(decomp) {
    const container = document.getElementById("modesContainer");
    container.innerHTML = "";

    const K = decomp.num_modes;
    document.getElementById("modesExtractedBadge").textContent = `${K} MVMFs Extracted`;

    const colors = ["#00d4ff", "#ff9800", "#10b981", "#f43f5e", "#a855f7", "#ec4899"];

    decomp.modes.forEach((mode, idx) => {
        const row = document.createElement("div");
        row.className = "mode-row";

        const canvasId = `modeCanvas_${idx}`;
        row.innerHTML = `
            <div class="mode-meta">
                <div class="mode-num">MVMF ${mode.mode_index}</div>
                <div class="mode-freq">fc = ${mode.center_freq_hz} Hz</div>
                <div class="mode-energy">Energy: ${mode.relative_energy_pct}%</div>
            </div>
            <div class="mode-canvas-box">
                <canvas id="${canvasId}" height="48"></canvas>
            </div>
        `;
        container.appendChild(row);

        // Draw mode trace
        setTimeout(() => {
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;
            const ctx = canvas.getContext("2d");
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * window.devicePixelRatio;
            canvas.height = 48 * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

            const W = rect.width;
            const H = 48;
            const vals = mode.values;
            const T = vals.length;

            ctx.strokeStyle = colors[idx % colors.length];
            ctx.lineWidth = 1.4;
            ctx.beginPath();

            const xStep = W / (T - 1);
            for (let t = 0; t < T; t++) {
                const x = t * xStep;
                const y = H / 2 - vals[t] * 6.0;
                if (t === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }, 30);
    });
}

function renderDiagnosticResults(pred, biomarkers) {
    const badge = document.getElementById("diagnosisBadge");
    const title = document.getElementById("predClassTitle");
    const sub = document.getElementById("predConfidenceText");
    const icon = document.getElementById("badgeIcon");

    title.textContent = pred.class_name;
    sub.textContent = `Diagnostic Confidence: ${pred.confidence}% via Energy-Weighted KNN`;

    if (pred.label === 0) {
        icon.textContent = "✓";
        icon.style.background = "rgba(16, 185, 129, 0.2)";
        icon.style.color = "#10b981";
    } else if (pred.label === 1) {
        icon.textContent = "⚡";
        icon.style.background = "rgba(245, 158, 11, 0.2)";
        icon.style.color = "#f59e0b";
    } else {
        icon.textContent = "🧠";
        icon.style.background = "rgba(239, 68, 68, 0.2)";
        icon.style.color = "#ef4444";
    }

    // Probabilities
    const probs = pred.probabilities;
    document.getElementById("probControlVal").textContent = `${probs["Typical Control"]}%`;
    document.getElementById("probControlBar").style.width = `${probs["Typical Control"]}%`;

    document.getElementById("probAdhdVal").textContent = `${probs["ADHD"]}%`;
    document.getElementById("probAdhdBar").style.width = `${probs["ADHD"]}%`;

    document.getElementById("probIddVal").textContent = `${probs["IDD"]}%`;
    document.getElementById("probIddBar").style.width = `${probs["IDD"]}%`;

    // Biomarkers
    document.getElementById("bmTbr").textContent = biomarkers.theta_beta_ratio.toFixed(2);
    document.getElementById("bmSlowFast").textContent = biomarkers.slow_fast_ratio.toFixed(2);
    document.getElementById("bmSampEn").textContent = biomarkers.sample_entropy.toFixed(3);
    document.getElementById("bmModesK").textContent = Object.keys(probs).length > 0 ? "Adaptive" : "--";
}

function renderEnergyChart(modes) {
    const ctx = document.getElementById("energyChart").getContext("2d");
    const labels = modes.map(m => `MVMF ${m.mode_index} (${m.center_freq_hz} Hz)`);
    const data = modes.map(m => m.relative_energy_pct);

    if (energyChartInstance) {
        energyChartInstance.destroy();
    }

    energyChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Relative Energy (%)",
                data: data,
                backgroundColor: ["#00d4ff", "#ff9800", "#10b981", "#f43f5e", "#a855f7", "#ec4899"],
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#9ca3af", font: { family: "Outfit", size: 10 } }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: "#9ca3af", font: { family: "Outfit", size: 9 } }
                }
            }
        }
    });
}

function resetDiagnosticBadges() {
    document.getElementById("predClassTitle").textContent = "Awaiting Analysis";
    document.getElementById("predConfidenceText").textContent = "Click 'Run SMVMD & Diagnose'";
    document.getElementById("badgeIcon").textContent = "?";
    document.getElementById("badgeIcon").style.background = "rgba(255, 255, 255, 0.08)";
    document.getElementById("badgeIcon").style.color = "#fff";

    document.getElementById("probControlVal").textContent = "0.0%";
    document.getElementById("probControlBar").style.width = "0%";
    document.getElementById("probAdhdVal").textContent = "0.0%";
    document.getElementById("probAdhdBar").style.width = "0%";
    document.getElementById("probIddVal").textContent = "0.0%";
    document.getElementById("probIddBar").style.width = "0%";

    document.getElementById("bmTbr").textContent = "--";
    document.getElementById("bmSlowFast").textContent = "--";
    document.getElementById("bmSampEn").textContent = "--";
    document.getElementById("bmModesK").textContent = "--";
}

async function runExperiment(expName) {
    const box = document.getElementById("experimentOutput");
    const txt = document.getElementById("expResultText");
    box.style.display = "block";
    txt.innerHTML = `<span style="color: #00d4ff;">Executing ${expName.toUpperCase()} experiment pipeline across full cohort... (Please wait ~5-10s)</span>`;

    try {
        const res = await fetch(`/api/run_paper_experiments?experiment=${expName}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Experiment failed");

        let html = `<strong>Experiment Completed: ${data.experiment}</strong><br><br>`;
        if (expName === "adhd") {
            html += `Primary KNN Accuracy (5-Fold CV): <span style="color: #10b981; font-weight: bold;">${data.primary_accuracy}%</span><br>`;
            html += `Sensitivity (Recall): <strong>${data.sensitivity}%</strong> | Specificity: <strong>${data.specificity}%</strong><br>`;
            html += `F1-Score: <strong>${data.f1_score}%</strong><br>`;
            html += `Subject-Independent LOSO Accuracy: <span style="color: #00d4ff; font-weight: bold;">${data.loso_accuracy}%</span><br>`;
        } else if (expName === "idd") {
            html += `Scenario 1 (Resting-State) Accuracy: <span style="color: #10b981; font-weight: bold;">${data.scenario_1_acc}%</span><br>`;
            html += `Scenario 2 (Fast 1.5s Epochs) Accuracy: <span style="color: #10b981; font-weight: bold;">${data.scenario_2_acc}%</span><br>`;
            html += `Scenario 3 (Noise Robustness) Accuracy: <span style="color: #10b981; font-weight: bold;">${data.scenario_3_acc}%</span><br>`;
        } else if (expName === "unified") {
            html += `Unified 3-Class Diagnosis Accuracy: <span style="color: #10b981; font-weight: bold;">${data.accuracy}%</span><br>`;
            html += `Macro F1-Score: <strong>${data.macro_f1}%</strong> | Cohen's Kappa: <strong>${data.kappa}</strong><br>`;
            html += `Multi-Class ROC-AUC: <span style="color: #00d4ff; font-weight: bold;">${data.roc_auc}%</span><br>`;
        }
        txt.innerHTML = html;
    } catch (err) {
        txt.innerHTML = `<span style="color: #ef4444;">Error running experiment: ${err.message}</span>`;
    }
}
