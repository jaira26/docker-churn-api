const form = document.getElementById("customer-form");
const result = document.getElementById("result");
const figure = document.getElementById("result-figure");
const caption = document.getElementById("result-caption");
const fill = document.getElementById("meter-fill");
const button = form.querySelector("button");

function collectPayload() {
  const payload = {};
  for (const el of form.elements) {
    if (!el.name) continue;
    payload[el.name] = el.type === "number" ? Number(el.value) : el.value;
  }
  return payload;
}

function describeError(body) {
  if (Array.isArray(body.detail)) {
    return body.detail
      .map((d) => `${d.loc[d.loc.length - 1]}: ${d.msg}`)
      .join("; ");
  }
  return body.detail || "The request failed.";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  button.disabled = true;
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectPayload()),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(describeError(body));

    const pct = Math.round(body.churn_probability * 100);
    result.className = `result ${body.will_churn ? "high" : "low"}`;
    figure.textContent = `${pct}%`;
    caption.textContent = body.will_churn
      ? "Likely to churn. Above the decision threshold."
      : "Likely to stay. Below the decision threshold.";
    fill.style.width = `${pct}%`;
  } catch (err) {
    result.className = "result error";
    figure.textContent = "--";
    caption.textContent = `Check not run. ${err.message}`;
    fill.style.width = "0";
  } finally {
    button.disabled = false;
  }
});
