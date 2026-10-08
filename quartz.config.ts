import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

/**
 * Quartz 4 Configuration
 *
 * See https://quartz.jzhao.xyz/configuration for more information.
 */
const config: QuartzConfig = {
  configuration: {
    pageTitle: "Study Notes",
    pageTitleSuffix: " — notes.swasilewski.com",
    enableSPA: true,
    enablePopovers: true,
    analytics: null,
    locale: "en-US",
    baseUrl: "notes.swasilewski.com",
    ignorePatterns: ["private", "templates", ".obsidian", "VAULT-INSTRUCTIONS.md"],
    defaultDateType: "modified",
    theme: {
      fontOrigin: "googleFonts",
      cdnCaching: true,
      typography: {
        // Archivo is also loaded with its width axis in Head.tsx, so headings can run condensed.
        header: { name: "Archivo", weights: [400, 600, 700] },
        body: { name: "Archivo", weights: [400, 500, 600], includeItalic: true },
        code: { name: "IBM Plex Mono", weights: [400, 500, 600] },
      },
      // Teenage Engineering panel look (picked 2026-10-08): warm grey ground, ink, one orange.
      colors: {
        lightMode: {
          light: "#e4e3de",
          lightgray: "#c9c7c0",
          gray: "#6b6a65",
          darkgray: "#2a2a28",
          dark: "#141414",
          secondary: "#b33a0c",
          tertiary: "#ff5a1f",
          highlight: "rgba(20, 20, 20, 0.05)",
          textHighlight: "#ff5a1f40",
        },
        darkMode: {
          light: "#141413",
          lightgray: "#33322e",
          gray: "#9a988f",
          darkgray: "#d6d4cd",
          dark: "#f4f3ef",
          secondary: "#ff7a45",
          tertiary: "#ff5a1f",
          highlight: "rgba(255, 255, 255, 0.05)",
          textHighlight: "#ff5a1f55",
        },
      },
    },
  },
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({
        priority: ["frontmatter", "git", "filesystem"],
      }),
      Plugin.SyntaxHighlighting({
        theme: {
          light: "github-light",
          dark: "github-dark",
        },
        keepBackground: false,
      }),
      Plugin.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: false }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents(),
      Plugin.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Description(),
      Plugin.Latex({ renderEngine: "katex" }),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.ContentIndex({
        enableSiteMap: true,
        enableRSS: true,
      }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.Favicon(),
      Plugin.NotFoundPage(),
      // Comment out CustomOgImages to speed up build time
      Plugin.CustomOgImages(),
    ],
  },
}

export default config
