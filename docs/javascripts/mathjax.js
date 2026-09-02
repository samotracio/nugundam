window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
    tags: "ams",
    macros: {
      rp: "r_{\\mathrm{p}}",
      wp: "w_{\\mathrm{p}}",
      pimax: "\\pi_{\\max}",
      dd: "\\mathrm{DD}",
      dr: "\\mathrm{DR}",
      rr: "\\mathrm{RR}"
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
