# Website

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

## Theme Redesign TODO

Redesigning the site to match the Logbook Hugo theme aesthetic - a clean, editorial/magazine style with minimalist design.

### Design Tasks

- [x] Update color scheme to neutral/minimalist (white background, dark text)
- [x] Update typography to clean sans-serif with clear hierarchy
- [x] Customize navbar - Blog-first layout with clean branding
- [x] Redesign footer with multi-column layout (Content, Social, Topics)
- [x] Create custom blog post card layout with enhanced styling
- [x] Add post metadata display (author, date, tags with pill styling)
- [x] Implement card-based blog listing with hover effects
- [x] Fix blog page width to match main/docs pages (1400px)
- [x] Add featured images and descriptions to blog posts
- [x] Add sidebar with search, author bio, categories, tags, latest posts
- [x] Implement author bio section for blog posts
- [x] Add social media sharing buttons
- [ ] Test responsive design on mobile and tablet
- [ ] Improve blog post card hover effects and animations
- [ ] Add reading time estimates to blog posts
- [ ] Implement table of contents for long blog posts
- [ ] Add dark mode toggle with smooth transitions
- [ ] Optimize featured image loading and sizing
- [ ] Add breadcrumb navigation
- [ ] Implement related posts section
- [ ] Add newsletter subscription form

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
