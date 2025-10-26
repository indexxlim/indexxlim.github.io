# Website

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

## Theme Redesign TODO

Redesigning the site to match the Logbook Hugo theme aesthetic - a clean, editorial/magazine style with minimalist design.

### Design Tasks

- [x] Update color scheme to neutral/minimalist (white background, dark text)
- [x] Update typography to clean sans-serif with clear hierarchy
- [ ] Customize navbar - horizontal layout with logo on left
- [ ] Create custom blog post card layout with large featured images
- [ ] Add post metadata display (author, date, categories, tags)
- [ ] Implement grid/card-based blog listing layout
- [ ] Add sidebar with search, author bio, categories, tags, latest posts
- [ ] Redesign footer with multi-column layout (logo, links, social, newsletter)
- [ ] Add featured images support for blog posts
- [ ] Implement author bio section for blog posts
- [ ] Add social media links and sharing buttons
- [ ] Test responsive design on mobile and tablet

### Design Reference

Based on [Logbook Hugo Theme](https://logbook-hugo.vercel.app) features:
- Clean editorial/magazine style
- Minimalist color palette (neutral backgrounds, dark text)
- Large featured images in blog cards
- Multi-column layout with rich sidebar
- Clear typography hierarchy
- Multi-column footer

## Installation

```bash
yarn
```

## Local Development

```bash
yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Deployment

Using SSH:

```bash
USE_SSH=true yarn deploy
```

Not using SSH:

```bash
GIT_USER=<Your GitHub username> yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
