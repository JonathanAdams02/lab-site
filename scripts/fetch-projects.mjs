// Fetches the unit's research projects from FRIS (the Flemish research
// portal, https://www.researchportal.be) and saves them to data/projects.json.
// Runs weekly on GitHub (see .github/workflows/update-projects.yml), or by hand:
//   node scripts/fetch-projects.mjs
//
// FRIS data are open data — source: Flemish government, Department of
// Economy, Science and Innovation.

import fs from "node:fs";
import path from "node:path";
import { XMLParser } from "fast-xml-parser";
import { load as loadYaml } from "js-yaml";

const ROOT = process.cwd();
const OUT = path.join(ROOT, "data", "projects.json");
const DEBUG_DIR = path.join(ROOT, "fris-debug");
const BASE = (process.env.FRIS_BASE || "https://frisr4.researchportal.be/ws").replace(/\/$/, "");
const PAGE_SIZE = 50;

const settingsFile = path.join(ROOT, "content", "projects-settings.yml");
const settings = fs.existsSync(settingsFile) ? loadYaml(fs.readFileSync(settingsFile, "utf8")) || {} : {};
const unitName = String(settings.fris_unit_name || "Neuropsychiatry").trim();
const unitUuidSetting = String(settings.fris_unit_uuid || "").trim();
const parentWords = (settings.fris_parent_keywords || ["Leuven"]).map((w) => String(w).toLowerCase());

const parser = new XMLParser({
  ignoreAttributes: false,
  attributeNamePrefix: "@",
  removeNSPrefix: true,
  textNodeName: "#text",
  parseTagValue: false,
  parseAttributeValue: false,
  isArray: (name) =>
    ["project", "organisation", "text", "participant", "projectFunding", "projectOrganisation",
     "fundingOrganisationAssociation", "keyword", "source", "type", "discipline"].includes(name),
});

let debugCount = 0;
function debugSave(name, text) {
  fs.mkdirSync(DEBUG_DIR, { recursive: true });
  fs.writeFileSync(path.join(DEBUG_DIR, `${String(++debugCount).padStart(2, "0")}-${name}`), text);
}

// ---------- SOAP plumbing (FRIS pages through results with a session cookie) ----------
const cookies = new Map();
async function soap(service, operation, criteriaXml, label) {
  const envelope = `<?xml version="1.0" encoding="UTF-8"?>
<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:fris="http://fris.ewi.be/" xmlns:crit="http://fris.ewi.be/criteria">
  <soapenv:Header/>
  <soapenv:Body>
    <fris:${operation}>
      <criteria>${criteriaXml}</criteria>
    </fris:${operation}>
  </soapenv:Body>
</soapenv:Envelope>`;
  const headers = { "Content-Type": "text/xml; charset=utf-8", SOAPAction: '""' };
  if (cookies.size) headers.Cookie = [...cookies].map(([k, v]) => `${k}=${v}`).join("; ");

  let res, text;
  for (let attempt = 1; ; attempt++) {
    res = await fetch(`${BASE}/${service}`, { method: "POST", headers, body: envelope });
    text = await res.text();
    if (res.status < 500 || /<faultstring>/.test(text) || attempt >= 4) break;
    await new Promise((r) => setTimeout(r, 3000 * attempt));
  }
  for (const c of res.headers.getSetCookie?.() || []) {
    const [pair] = c.split(";");
    const i = pair.indexOf("=");
    if (i > 0) cookies.set(pair.slice(0, i).trim(), pair.slice(i + 1).trim());
  }
  debugSave(`${label}.xml`, text);
  const fault = text.match(/<faultstring>([\s\S]*?)<\/faultstring>/);
  if (fault || !res.ok) {
    throw new Error(`FRIS ${service}.${operation} failed (${res.status}): ${fault ? fault[1] : text.slice(0, 300)}`);
  }
  const doc = parser.parse(text);
  const body = doc?.Envelope?.Body || {};
  const response = body[`${operation}Response`] || {};
  return response.queryResult || response;
}

const windowXml = (page, orderId) => `
        <crit:window>
          <crit:pageSize>${PAGE_SIZE}</crit:pageSize>
          <crit:pageNumber>${page}</crit:pageNumber>
          <crit:orderings><crit:order><crit:id>${orderId}</crit:id><crit:locale>en</crit:locale><crit:direction>DESCENDING</crit:direction></crit:order></crit:orderings>
        </crit:window>`;

const esc = (s) => String(s).replace(/[<>&]/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" })[c]);
const arr = (x) => (x === undefined || x === null ? [] : Array.isArray(x) ? x : [x]);

// Pick the English text from a FRIS LocalizedString, falling back to Dutch / anything.
function localized(node) {
  const texts = arr(node?.texts?.text);
  const get = (loc) => texts.find((t) => String(t?.["@locale"] || "").toLowerCase().startsWith(loc));
  const t = get("en") || get("nl") || texts[0];
  return clean(typeof t === "string" ? t : t?.["#text"] || "");
}
function clean(s) {
  return String(s || "")
    .replace(/<br\s*\/?>/gi, "\n").replace(/<\/p>/gi, "\n\n").replace(/<[^>]+>/g, "")
    .replace(/&nbsp;/g, " ").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;/g, "'")
    .replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
}

// Fetch every page of a query (stops when all results are in, or a page repeats).
async function fetchAll(service, operation, criteriaInner, label, orderId, entity) {
  const all = [];
  const seen = new Set();
  for (let page = 0; page < 100; page++) {
    const result = await soap(service, operation, windowXml(page, orderId) + criteriaInner, `${label}-page${page}`);
    const items = arr(result?.[entity]);
    const total = Number(result?.["@totalResults"] ?? result?.totalResults ?? NaN);
    const fresh = items.filter((it) => !seen.has(it["@uuid"]));
    fresh.forEach((it) => seen.add(it["@uuid"]));
    all.push(...fresh);
    console.log(`  ${label}: page ${page} → ${items.length} items (total reported: ${Number.isNaN(total) ? "?" : total})`);
    if (!items.length || !fresh.length || (!Number.isNaN(total) && all.length >= total) || items.length < PAGE_SIZE) break;
  }
  return all;
}

// ---------- 1. Find the unit ----------
async function findUnit() {
  if (unitUuidSetting) return { uuid: unitUuidSetting, name: unitName };
  console.log(`Looking up "${unitName}" in FRIS…`);
  const orgs = await fetchAll(
    "OrganisationServiceFRIS", "getOrganisations",
    `<crit:name><crit:search>${esc(unitName)}</crit:search><crit:locale>en</crit:locale></crit:name>`,
    "organisations", "entity.created", "organisation"
  );
  const candidates = orgs.map((o) => {
    const xmlish = JSON.stringify(o).toLowerCase();
    return {
      uuid: o["@uuid"],
      name: localized(o.name),
      matchesName: localized(o.name).toLowerCase() === unitName.toLowerCase(),
      matchesParent: parentWords.some((w) => xmlish.includes(w)),
      provider: o.dataProvider?.name ? localized(o.dataProvider.name) : JSON.stringify(o.dataProvider || "").slice(0, 80),
    };
  });
  console.log("Candidates:\n" + candidates.map((c) => `  - ${c.name} | ${c.uuid} | provider: ${c.provider} | name match: ${c.matchesName} | parent match: ${c.matchesParent}`).join("\n"));
  let best = candidates.filter((c) => c.matchesName && c.matchesParent);
  if (best.length !== 1) best = candidates.filter((c) => c.matchesName);
  if (best.length !== 1) {
    throw new Error(
      `Could not pick a single FRIS unit called "${unitName}" (${best.length} matches). ` +
      `Copy the right UUID from the list above into "FRIS unit ID" under Project list settings in Pages CMS.`
    );
  }
  console.log(`Using unit ${best[0].name} (${best[0].uuid}). Tip: save this ID in Project list settings to skip the lookup.`);
  return best[0];
}

// ---------- 2. Fetch its projects ----------
const unit = await findUnit();
console.log(`Fetching projects for ${unit.uuid}…`);
const raw = await fetchAll(
  "ProjectServiceFRIS", "getProjects",
  `<crit:associatedOrganisations><crit:identifier>${esc(unit.uuid)}</crit:identifier></crit:associatedOrganisations>`,
  "projects", "entity.created", "project"
);

const KUL_CODE_RE = /^[0-9][A-Z][0-9]{6}$/; // e.g. 3M230323, 3E220123
const isoDate = (d) => (d ? String(d).slice(0, 10) : "");
const projects = raw
  .filter((p) => String(p.hidden) !== "true" && (!p.state || p.state === "PUBLIC"))
  .map((p) => {
    const participants = arr(p.participants?.participant).map((pt) => {
      const nm = pt.person?.name || pt.name || {};
      const name = [nm.firstName, nm.lastName].filter(Boolean).join(" ").trim();
      const role = localized(pt.associationType?.description) || pt.associationType?.["@term"] || "";
      return { name, role };
    }).filter((x) => x.name);

    // The list and its items share the tag name "projectFunding".
    const fundingItems = arr(p.projectFunding).flatMap((x) => (x && x.projectFunding ? arr(x.projectFunding) : [x]));
    const funding = fundingItems.map((f) => {
      const fc = f.fundingCode || {};
      const funders = arr(fc.fundingOrganisationAssociations?.fundingOrganisationAssociation)
        .map((a) => localized(a.organisation?.name)).filter(Boolean);
      return { code: localized(fc.description) || fc.acronym || "", funders };
    }).filter((f) => f.code || f.funders.length);

    // All identifiers FRIS gives for the project. The KU Leuven research
    // portal addresses projects by their KU Leuven project code (e.g. 3M230323),
    // which is not the FRIS dataProviderId — so look for a code of that shape.
    const identifiers = [
      ...arr(p.sources?.source).map((x) => ({ kind: "source", authority: x?.["@authority"] || "", value: String(x?.["#text"] ?? x ?? "").trim() })),
      ...arr(p.fundingIdentifiers?.fundingIdentifier).map((x) => ({ kind: "funding", authority: x?.["@authority"] || "", value: String(x?.["#text"] ?? x ?? "").trim() })),
    ].filter((x) => x.value);
    const kulCode = identifiers.map((x) => x.value).find((v) => KUL_CODE_RE.test(v));
    const kuleuvenUrl = kulCode ? `https://research.kuleuven.be/portal/en/project/${encodeURIComponent(kulCode)}` : "";

    return {
      uuid: p["@uuid"],
      title: localized(p.name).replace(/^["“'\s]+|["”'\s]+$/g, ""),
      acronym: p.acronym || "",
      abstract: localized(p.projectAbstract),
      start: isoDate(p.startDate),
      end: isoDate(p.endDate),
      funding,
      participants,
      url: kuleuvenUrl || (p.homepage ? String(p.homepage) : ""),
      fris_id: String(p.dataProviderId || ""),
      fris_uri: p["@uri"] || "",
      identifiers,
      keywords: arr(p.keywords?.keyword).map((k) => clean(typeof k === "string" ? k : k?.["#text"])).filter(Boolean),
    };
  })
  .filter((p) => p.title)
  .sort((a, b) => (b.start || "").localeCompare(a.start || "") || a.title.localeCompare(b.title));

// Show a few examples in the log, to help check identifiers and links.
for (const p of projects.slice(0, 3)) {
  console.log(`  e.g. "${p.title.slice(0, 60)}" → link: ${p.url || "(none)"} | ids: ${p.identifiers.map((x) => `${x.kind}:${x.authority}:${x.value}`).join(", ") || "(none)"} | uri: ${p.fris_uri || "(none)"}`);
}
const linked = projects.filter((p) => p.url).length;
console.log(`${linked} of ${projects.length} projects have a link to their portal page.`);

// ---------- 3. Save (keep the old timestamp when nothing changed) ----------
let previous = null;
try { previous = JSON.parse(fs.readFileSync(OUT, "utf8")); } catch {}
const unchanged = previous && JSON.stringify(previous.projects) === JSON.stringify(projects);
fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, JSON.stringify({
  source: "FRIS — Flemish government, Department of Economy, Science and Innovation",
  unit: { name: unit.name, uuid: unit.uuid },
  updated: unchanged ? previous.updated : new Date().toISOString(),
  projects,
}, null, 2) + "\n");
console.log(`${projects.length} projects saved to data/projects.json${unchanged ? " (no changes)" : ""}.`);
