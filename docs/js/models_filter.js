/**
 * Sortable + filterable models table.
 * Runs inside MkDocs Material's document$ observable so it
 * re-executes on soft navigations (instant loading).
 */
document$.subscribe(function () {
  const table = document.getElementById("models-table");
  if (!table) return;

  // ── Tablesort ──────────────────────────────────────────────────
  if (typeof Tablesort !== "undefined") {
    new Tablesort(table);
  }

  // ── State ──────────────────────────────────────────────────────
  const allRows = Array.from(table.querySelectorAll("tbody tr"));
  let activeModalities = new Set(["all"]);
  let searchTerm = "";

  function applyFilters() {
    let visible = 0;
    allRows.forEach(function (row) {
      const modality = row.dataset.modality || "";
      const text = row.textContent.toLowerCase();
      const modalityMatch =
        activeModalities.has("all") || activeModalities.has(modality);
      const searchMatch =
        searchTerm === "" || text.includes(searchTerm);
      const show = modalityMatch && searchMatch;
      row.style.display = show ? "" : "none";
      if (show) visible++;
    });

    const counter = document.getElementById("models-visible-count");
    if (counter) {
      counter.textContent = visible + " model" + (visible !== 1 ? "s" : "");
    }
  }

  // ── Multiselect modality filter buttons ────────────────────────
  const filterBtns = document.querySelectorAll(".modality-filter");
  filterBtns.forEach(function (btn) {
    btn.addEventListener("click", function () {
      const filter = btn.dataset.filter;

      if (filter === "all") {
        activeModalities = new Set(["all"]);
      } else {
        activeModalities.delete("all");
        if (activeModalities.has(filter)) {
          activeModalities.delete(filter);
          if (activeModalities.size === 0) {
            activeModalities = new Set(["all"]);
          }
        } else {
          activeModalities.add(filter);
        }
      }

      // Reflect active state on buttons
      filterBtns.forEach(function (b) {
        const isActive =
          b.dataset.filter === "all"
            ? activeModalities.has("all")
            : activeModalities.has(b.dataset.filter);
        b.classList.toggle("active", isActive);
      });

      applyFilters();
    });
  });

  // ── Search box ─────────────────────────────────────────────────
  const searchInput = document.getElementById("models-search");
  if (searchInput) {
    searchInput.addEventListener("input", function () {
      searchTerm = searchInput.value.toLowerCase().trim();
      applyFilters();
    });
  }

  // Initial render
  applyFilters();
});
