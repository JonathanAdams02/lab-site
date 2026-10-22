# Neuropsychiatry · KU Leuven — website

The group's website. Its look matches the original design file. All text, people, publications, projects and studies can be edited through simple forms in [Pages CMS](https://pagescms.org), with no code involved.

```
content/      ← everything editors change (Pages CMS edits these files)
  pages/        texts for each page (home, about, people, …)
  people/       one file per team member
  publications/ one file per paper
  projects/     one file per project
  studies/      one file per recruiting study
  settings.yml  emails, logos, atlas address, footer links
  publications.yml  settings for the automatic publication list
data/publications.json  ← publication list fetched from OpenAlex (updated automatically)
scripts/      ← the OpenAlex fetch script
media/        ← photos uploaded through Pages CMS
src/          ← design: page templates, stylesheet, fonts, logos
.pages.yml    ← defines the editing forms
.github/      ← builds and publishes the site automatically
```

---

## 1 · Put it on GitHub (once)

1. **Create a repository** on GitHub. It can be public or private (private needs a paid GitHub plan for Pages). Ideally, create it under a GitHub *organisation* for the lab rather than a personal account, so the site survives people leaving.
2. **Upload all files** from this folder, **including the hidden ones**: `.github/` and `.pages.yml`. On a Mac, press <kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>.</kbd> in Finder to show hidden files before dragging them into GitHub's "Add file → Upload files" page. GitHub Desktop or `git push` also works. Make sure the branch is called `main`.
3. In the repository, go to **Settings → Pages**. Under *Build and deployment → Source*, choose **GitHub Actions**.
4. Go to the **Actions** tab, open *Build and publish site* and click **Run workflow**. Uploads made before step 3 couldn't publish yet, which is why this manual run is needed. After about a minute the site is live at `https://<your-account>.github.io/<repo-name>/`.

From then on, every change saved (through Pages CMS or otherwise) republishes the site automatically within 1–2 minutes.

## 2 · Connect your domain (once)

1. In **Settings → Pages → Custom domain**, enter your domain (e.g. `neuropsychiatry-lab.be`) and save.
2. At the company where you bought the domain, add these DNS records:

   | Type  | Name / Host | Value |
   |-------|-------------|-------|
   | A     | `@`         | `185.199.108.153` |
   | A     | `@`         | `185.199.109.153` |
   | A     | `@`         | `185.199.110.153` |
   | A     | `@`         | `185.199.111.153` |
   | CNAME | `www`       | `<your-account>.github.io` |

3. Wait for the DNS check in Settings → Pages to pass (minutes to a few hours), then tick **Enforce HTTPS**.
4. Optional but recommended: verify the domain under your GitHub account/organisation settings → *Pages*, so nobody else can claim it.

Links, images and stylesheets adjust themselves automatically, whether the site runs on `github.io/<repo-name>/` or on your own domain.

## 3 · Set up editing for colleagues (once)

1. Go to **[app.pagescms.org](https://app.pagescms.org)**, sign in with GitHub, and give Pages CMS access to the repository when asked.
2. Open the repository there. The editing forms appear in the sidebar.
3. Invite colleagues. Pages CMS can invite people by email (they then don't need a GitHub account), or you can add them as collaborators on the GitHub repository.

## 4 · Automatic publication list

The publication list on *Research output* fills itself from [OpenAlex](https://openalex.org), a free, open database of scholarly papers.

1. In Pages CMS, open each researcher under **People** and fill in their **ORCID iD**. For someone without an ORCID, use the *OpenAlex author ID* field instead.
2. Saving starts an update straight away. You can also start one by hand in the GitHub **Actions** tab → *Update publications* → **Run workflow**.
3. After that it checks for new papers **every Monday**. When it finds some, it saves them and republishes the site, with no one needing to do anything.

Fine-tuning, in Pages CMS → **Publication list settings**:
- **From year**: how far back the list goes.
- **Leuven affiliation only** (on by default): only papers where the lab member listed a Leuven affiliation, so a new member's earlier work elsewhere stays out.
- **Publication types**: articles, reviews, chapters and so on.
- **Hide these publications**: paste the DOI of any paper that shouldn't be there.

Anything OpenAlex misses can be added under **Extra publications (manual)**. A manual entry with the same DOI replaces the automatic one, which is useful for correcting a title or journal name. **Delete the placeholder publications there** once the real list appears.

Lab members are shown in **bold** in author lists. Because the update runs on GitHub's servers, nothing needs to be running on anyone's computer.

> **Optional:** OpenAlex gives faster, more reliable access to users who identify themselves. In GitHub, go to **Settings → Secrets and variables → Actions → Variables** and add `OPENALEX_MAILTO` with a lab email address. If OpenAlex asks for an API key in future, add it under *Secrets* as `OPENALEX_API_KEY`.
>
> GitHub pauses scheduled jobs in repositories with no activity for 60 days. If the list stops updating, open Actions → *Update publications* and re-enable it, or just save any change in Pages CMS.

---

## Editing guide (for everyone)

Log in at **app.pagescms.org**, open the site, then:

| To… | Go to… |
|---|---|
| Add or update a **team member** | **People** → *Add an entry* (or click a name). Choose the group, upload a square photo, save. |
| Get someone's **publications** listed | **People** → their entry → fill in the **ORCID iD**. Papers appear automatically (see section 4). |
| Add a publication OpenAlex missed | **Extra publications (manual)** → *Add an entry*. |
| Hide a wrong publication | **Publication list settings** → *Hide these publications* → paste its DOI. |
| Add or update a **project** | **Projects**. Use *Position in the list* to set the order. |
| Open or close a **recruiting study** | **Recruiting studies**. Untick *Currently recruiting* to hide a study without deleting it. |
| Change **page texts** (intro, About, Contact…) | **Page texts** → the page. |
| Change **emails, logos, footer links, atlas address** | **Site settings** |

Click **Save**. The live site updates about 1–2 minutes later (refresh the page). Every save is kept in the history, so nothing is ever lost and any change can be undone on GitHub.

**Tips**
- People without a photo show their initials. Square photos of at least 400×400 px look best.
- The *order* fields are optional: lower numbers come first, and empty ones go last.
- Leaving an optional text field empty hides that element on the site.

---

## For whoever maintains the design

- **Run it locally:** install [Node.js](https://nodejs.org) (v20+), then `npm install` and `npm start`, and open http://localhost:8080.
- **Built with** [Eleventy](https://www.11ty.dev). The templates are in `src/*.njk` and `src/_includes/`, all styling is in `src/assets/css/site.css`, and `src/_data/lab.js` reads the `content/` folder.
- **Fonts** (Libre Franklin, Source Serif 4) are self-hosted in `src/assets/fonts/`, so visitors' browsers never contact Google (good for GDPR).
- **Adding a team group:** add it to the People page's group list (`content/pages/people.md`) *and* to the `group` choices in `.pages.yml`.
- **Navigation menu:** `src/_data/navigation.json`.
