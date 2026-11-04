import { HtmlBasePlugin } from "@11ty/eleventy";

export default function (eleventyConfig) {
  // Rewrites links like /about/ to /<repo-name>/about/ when the site is not
  // (yet) on its own domain. The deploy workflow sets the prefix automatically.
  eleventyConfig.addPlugin(HtmlBasePlugin);

  // Escape anything editors type, unless a template marks it `| safe`.
  eleventyConfig.setNunjucksEnvironmentOptions({ autoescape: true });

  eleventyConfig.addPassthroughCopy({ "src/assets": "assets" });
  eleventyConfig.addPassthroughCopy({ media: "media" });
  // The Research Atlas, built by atlas/build.py, is published at /atlas/.
  eleventyConfig.addPassthroughCopy({ "atlas/atlas.html": "atlas/index.html" });

  // Rebuild when content changes during local preview.
  eleventyConfig.addWatchTarget("content/");

  // Email addresses are never written into the pages as plain text (spam
  // harvesters read the page source). They're stored reversed + base64 and
  // put back together in the browser when a visitor clicks — see email.njk.
  eleventyConfig.addFilter("encodeEmail", (address = "") =>
    Buffer.from(String(address).trim().split("").reverse().join(""), "utf8").toString("base64")
  );

    // "Ada Lovelace" -> "AL" (used when a person has no photo).
  eleventyConfig.addFilter("initials", (name = "") =>
    String(name)
      .split(/\s+/)
      .filter(Boolean)
      .filter((_, i, arr) => i === 0 || i === arr.length - 1)
      .map((w) => w[0].toUpperCase())
      .join("")
  );

  return {
    dir: { input: "src", includes: "_includes", data: "_data", output: "_site" },
    templateFormats: ["njk"],
    htmlTemplateEngine: "njk",
  };
}
