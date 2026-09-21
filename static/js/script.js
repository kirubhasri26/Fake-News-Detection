// Shows a "Analyzing news content..." loading indicator when the
// prediction form is submitted, so the page does not look frozen
// while the server processes the request.
document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("predictForm");
    if (!form) {
        return;
    }

    form.addEventListener("submit", function () {
        const analyzeBtn = document.getElementById("analyzeBtn");
        const loadingMessage = document.getElementById("loadingMessage");

        if (analyzeBtn) {
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = "Analyzing...";
        }
        if (loadingMessage) {
            loadingMessage.classList.remove("d-none");
        }
    });
});
