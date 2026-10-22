// One-time probe of the FRIS web services (https://frisr4.researchportal.be/ws/).
// Saves the service descriptions (WSDL/XSD) and the raw answers to a few test
// requests into the folder fris-probe/, so the real project importer can be
// written against actual data. Safe to run: it only reads public data.
//
//   node scripts/fris-probe.mjs

import fs from "node:fs";
import path from "node:path";

const BASE = "https://frisr4.researchportal.be/ws";
const OUT = path.resolve("fris-probe");
const UNIT_NAME = process.env.FRIS_UNIT_NAME || "Neuropsychiatry";
fs.mkdirSync(OUT, { recursive: true });

const log = [];
const note = (msg) => { console.log(msg); log.push(msg); };
const save = (name, text) => fs.writeFileSync(path.join(OUT, name), text);

async function get(url) {
  try {
    const res = await fetch(url);
    const text = await res.text();
    note(`GET ${url} -> ${res.status} (${text.length} chars)`);
    return text;
  } catch (e) {
    note(`GET ${url} -> ERROR ${e.message}`);
    return "";
  }
}

async function soap(service, name, body) {
  const envelope = `<?xml version="1.0" encoding="UTF-8"?>
<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:fris="http://fris.ewi.be/" xmlns:crit="http://fris.ewi.be/criteria">
  <soapenv:Header/>
  <soapenv:Body>
${body}
  </soapenv:Body>
</soapenv:Envelope>`;
  save(`${name}.request.xml`, envelope);
  try {
    const res = await fetch(`${BASE}/${service}`, {
      method: "POST",
      headers: { "Content-Type": "text/xml; charset=utf-8", SOAPAction: '""' },
      body: envelope,
    });
    const text = await res.text();
    save(`${name}.response.xml`, text);
    note(`POST ${service} [${name}] -> ${res.status} (${text.length} chars)`);
    return text;
  } catch (e) {
    note(`POST ${service} [${name}] -> ERROR ${e.message}`);
    return "";
  }
}

// 1. Service descriptions, plus every schema they import.
for (const svc of ["ProjectServiceFRIS", "OrganisationServiceFRIS"]) {
  const wsdl = await get(`${BASE}/${svc}?wsdl`);
  save(`${svc}.wsdl.xml`, wsdl);
  const seen = new Set();
  const queue = [...wsdl.matchAll(/schemaLocation="([^"]+)"/g)].map((m) => m[1]);
  while (queue.length) {
    const loc = queue.shift().replace(/&amp;/g, "&");
    const url = loc.startsWith("http") ? loc : `${BASE}/${svc}${loc.startsWith("?") ? "" : "/"}${loc}`;
    if (seen.has(url)) continue;
    seen.add(url);
    const xsd = await get(url);
    save(`${svc}.schema-${seen.size}.xsd.xml`, xsd);
    for (const m of xsd.matchAll(/schemaLocation="([^"]+)"/g)) queue.push(m[1]);
  }
}

// 2. Small test requests — several variants, since the exact criteria format
//    is what we're trying to learn. Failures (SOAP faults) are informative too.
const win = (size) => `<crit:window><crit:pageSize>${size}</crit:pageSize><crit:pageNumber>0</crit:pageNumber></crit:window>`;

await soap("OrganisationServiceFRIS", "org-by-name", `
    <fris:getOrganisations>
      <organisationCriteria>
        ${win(20)}
        <crit:name>${UNIT_NAME}</crit:name>
      </organisationCriteria>
    </fris:getOrganisations>`);

await soap("OrganisationServiceFRIS", "org-by-search", `
    <fris:getOrganisations>
      <organisationCriteria>
        ${win(20)}
        <crit:search><crit:value>${UNIT_NAME}</crit:value><crit:locale>en</crit:locale></crit:search>
      </organisationCriteria>
    </fris:getOrganisations>`);

await soap("ProjectServiceFRIS", "projects-by-search", `
    <fris:getProjects>
      <projectCriteria>
        ${win(5)}
        <crit:search><crit:value>"late life depression"</crit:value><crit:locale>en</crit:locale></crit:search>
      </projectCriteria>
    </fris:getProjects>`);

await soap("ProjectServiceFRIS", "projects-default", `
    <fris:getProjects>
      <projectCriteria>
        ${win(3)}
      </projectCriteria>
    </fris:getProjects>`);

await soap("ProjectServiceFRIS", "orderings", `
    <fris:getOrderings/>`);

// 3. If an organisation UUID turned up, try fetching that unit's projects.
const orgXml = ["org-by-name", "org-by-search"]
  .map((n) => { try { return fs.readFileSync(path.join(OUT, `${n}.response.xml`), "utf8"); } catch { return ""; } })
  .join("\n");
const uuids = [...new Set([...orgXml.matchAll(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi)].map((m) => m[0]))];
note(`Organisation UUIDs seen: ${uuids.slice(0, 10).join(", ") || "none"}`);
for (const [i, uuid] of uuids.slice(0, 3).entries()) {
  await soap("ProjectServiceFRIS", `projects-for-org-${i + 1}`, `
    <fris:getProjects>
      <projectCriteria>
        ${win(5)}
        <crit:associatedOrganisations><crit:identifier>${uuid}</crit:identifier></crit:associatedOrganisations>
      </projectCriteria>
    </fris:getProjects>`);
}

save("_log.txt", log.join("\n") + "\n");
note(`Done — everything saved in ${OUT}`);
