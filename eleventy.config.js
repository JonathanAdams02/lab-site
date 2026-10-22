import { HtmlBasePlugin } from "@11ty/eleventy";

export default function (eleventyConfig) {
  // Rewrites links like /about/ to /<repo-name>/about/ when the site is not
  // (yet) on its own domain. The deploy workflow sets the prefix automatically.
  eleventyConfig.addPlugin(HtmlBasePlugin);

  // Escape anything editors type, unless a template marks it `| safe`.
  eleventyConfig.setNunjucksEnvironmentOptions({ autoescape: true });

  eleventyConfig.addPassthroughCopy({ "src/assets": "assets" });
  eleventyConfig.addPassthroughCopy({ media: "media" });

  // Rebuild when content changes during local preview.
  eleventyConfig.addWatchTarget("content/");

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
