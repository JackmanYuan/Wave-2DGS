document.querySelectorAll(".image-compare").forEach((comparison) => {
  const comparisonRange = comparison.querySelector(".compare-range");
  if (!comparisonRange) return;

  const updatePosition = () => {
    comparison.style.setProperty("--position", `${comparisonRange.value}%`);
    const leftLabel = comparison.dataset.leftLabel || "left image";
    comparisonRange.setAttribute(
      "aria-valuetext",
      `${comparisonRange.value}% ${leftLabel} visible`
    );
  };

  comparisonRange.addEventListener("input", updatePosition);
  updatePosition();
});
