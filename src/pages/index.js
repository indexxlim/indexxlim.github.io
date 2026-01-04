import {useEffect, useMemo, useState} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import {useAllPluginInstancesData} from '@docusaurus/useGlobalData';
import styles from './index.module.css';

const FALLBACK_AUTOROTATE_MS = 6500;

const formatDate = (dateString) => {
  if (!dateString) {
    return '';
  }
  const date = new Date(dateString);
  if (Number.isNaN(date.getTime())) {
    return dateString;
  }
  return new Intl.DateTimeFormat('en-US', {
    month: 'long',
    day: 'numeric',
    year: 'numeric',
  }).format(date);
};

const formatReadingTime = (readingTime) => {
  if (!readingTime) {
    return '';
  }
  const minutes = Math.max(1, Math.round(readingTime));
  return `${minutes} min read`;
};

const stripMarkdown = (value) =>
  value
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/!\[.*?\]\(.*?\)/g, ' ')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/#+\s/g, '')
    .replace(/<\/?[^>]+(>|$)/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

const getExcerpt = (value, maxLength = 160) => {
  if (!value) {
    return '';
  }
  const cleaned = stripMarkdown(value);
  if (cleaned.length <= maxLength) {
    return cleaned;
  }
  return `${cleaned.slice(0, maxLength).trim()}...`;
};

function MetaLine({author, date, tags, readingTime}) {
  const tagLabel = tags?.length ? tags.join(', ') : '';
  const formattedDate = formatDate(date);
  const formattedReadingTime = formatReadingTime(readingTime);

  return (
    <div className={styles.meta}>
      {author ? <span>{author}</span> : null}
      {author && formattedDate ? <span className={styles.metaDot}>·</span> : null}
      {formattedDate ? <span>{formattedDate}</span> : null}
      {formattedDate && tagLabel ? <span className={styles.metaDot}>·</span> : null}
      {tagLabel ? <span>{tagLabel}</span> : null}
      {formattedReadingTime ? (
        <>
          <span className={styles.metaDot}>·</span>
          <span>{formattedReadingTime}</span>
        </>
      ) : null}
    </div>
  );
}

export default function Home() {
  const blogInstances = useAllPluginInstancesData('docusaurus-plugin-content-blog');
  const blogPosts = blogInstances?.default?.blogPosts ?? [];

  const normalizedPosts = useMemo(
    () =>
      blogPosts.map((post) => {
        const {metadata, content} = post;
        const frontMatter = metadata?.frontMatter ?? {};
        const tags = metadata?.tags?.map((tag) => tag.label) ?? [];

        return {
          id: post.id,
          title: metadata?.title ?? 'Untitled',
          permalink: metadata?.permalink ?? '/blog',
          date: metadata?.date,
          author: metadata?.authors?.[0]?.name ?? '',
          tags,
          readingTime: metadata?.readingTime,
          image: frontMatter.image || frontMatter.cover_image || '',
          excerpt: getExcerpt(metadata?.description || content || ''),
        };
      }),
    [blogPosts],
  );

  const featuredPosts = normalizedPosts.slice(0, 3);
  const listPosts = normalizedPosts.slice(featuredPosts.length ? 1 : 0, 5);
  const recentPosts = normalizedPosts.slice(0, 3);
  const uniqueTags = Array.from(
    new Set(normalizedPosts.flatMap((post) => post.tags)),
  );
  const categories = uniqueTags.slice(0, 5);
  const tagList = uniqueTags.slice(0, 8);

  const [activeIndex, setActiveIndex] = useState(0);

  useEffect(() => {
    if (featuredPosts.length <= 1) {
      return undefined;
    }
    if (typeof window === 'undefined') {
      return undefined;
    }
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (mediaQuery.matches) {
      return undefined;
    }
    const id = window.setInterval(() => {
      setActiveIndex((prev) => (prev + 1) % featuredPosts.length);
    }, FALLBACK_AUTOROTATE_MS);
    return () => window.clearInterval(id);
  }, [featuredPosts.length]);

  useEffect(() => {
    if (activeIndex > featuredPosts.length - 1) {
      setActiveIndex(0);
    }
  }, [activeIndex, featuredPosts.length]);

  const activePost = featuredPosts[activeIndex];
  const hasSlides = featuredPosts.length > 0;
  const canSlide = featuredPosts.length > 1;

  return (
    <Layout title="Logbook" description="A calm, journal-style homepage layout">
      <main>
        <div className={styles.page}>
          <header className={styles.heroSection}>
            <div className={styles.heroIntro}>
              <span className={styles.heroKicker}>The Logbook</span>
              <Heading as="h1" className={styles.heroTitle}>
                Notes from a slow studio, collected every week.
              </Heading>
              <p className={styles.heroText}>
                A curated log of stories, sketches, and quiet observations. Browse the
                journal, explore categories, and settle into a calmer pace.
              </p>
              <div className={styles.heroActions}>
                <Link className={styles.primaryButton} to="/blog">
                  Read the journal
                </Link>
                <Link className={styles.secondaryButton} to="/docs/intro">
                  About the studio
                </Link>
              </div>
              <div className={styles.heroStats}>
                <div>
                  <span className={styles.heroStatNumber}>{normalizedPosts.length}</span>
                  <span className={styles.heroStatLabel}>Entries</span>
                </div>
                <div>
                  <span className={styles.heroStatNumber}>{categories.length}</span>
                  <span className={styles.heroStatLabel}>Collections</span>
                </div>
                <div>
                  <span className={styles.heroStatNumber}>
                    {Math.min(normalizedPosts.length, 4)}
                  </span>
                  <span className={styles.heroStatLabel}>Updates / mo</span>
                </div>
              </div>
            </div>

            <article
              className={styles.heroFeature}
              role="region"
              aria-roledescription="carousel"
              aria-label="Featured posts">
              {hasSlides ? (
                <>
                  <div
                    className={clsx(
                      styles.heroImage,
                      !activePost?.image && styles.heroImageFallback,
                    )}
                    style={
                      activePost?.image
                        ? {backgroundImage: `url(${activePost.image})`}
                        : undefined
                    }
                  />
                  <div className={styles.heroOverlay} aria-live="polite">
                    <span className={styles.heroBadge}>Featured Story</span>
                    <Heading as="h2" className={styles.heroFeatureTitle}>
                      {activePost?.title}
                    </Heading>
                    <MetaLine
                      author={activePost?.author}
                      date={activePost?.date}
                      tags={activePost?.tags}
                      readingTime={activePost?.readingTime}
                    />
                    <p className={styles.heroExcerpt}>{activePost?.excerpt}</p>
                    <Link className={styles.heroLink} to={activePost?.permalink || '/blog'}>
                      Continue reading →
                    </Link>
                  </div>
                  <div className={styles.heroControls}>
                    <button
                      className={styles.heroArrow}
                      type="button"
                      onClick={() =>
                        setActiveIndex((prev) =>
                          (prev - 1 + featuredPosts.length) % featuredPosts.length,
                        )
                      }
                      disabled={!canSlide}
                      aria-label="Previous featured post">
                      ‹
                    </button>
                    <div className={styles.heroDots}>
                      {featuredPosts.map((post, index) => (
                        <button
                          key={post.id}
                          className={clsx(
                            styles.heroDot,
                            index === activeIndex && styles.heroDotActive,
                          )}
                          type="button"
                          onClick={() => setActiveIndex(index)}
                          aria-label={`Go to slide ${index + 1}`}
                          aria-current={index === activeIndex ? 'true' : undefined}
                        />
                      ))}
                    </div>
                    <button
                      className={styles.heroArrow}
                      type="button"
                      onClick={() =>
                        setActiveIndex((prev) => (prev + 1) % featuredPosts.length)
                      }
                      disabled={!canSlide}
                      aria-label="Next featured post">
                      ›
                    </button>
                  </div>
                </>
              ) : (
                <div className={styles.heroEmpty}>
                  <p>No featured posts yet. Add a blog post to get started.</p>
                </div>
              )}
            </article>
          </header>

          <section className={styles.contentSection}>
            <div className={styles.postsArea}>
              <div className={styles.sectionHeading}>
                <div>
                  <span className={styles.sectionKicker}>Latest Stories</span>
                  <Heading as="h2" className={styles.sectionTitle}>
                    The newest entries from the journal.
                  </Heading>
                </div>
                <Link className={styles.sectionLink} to="/blog">
                  View all posts
                </Link>
              </div>
              <div className={styles.postsGrid}>
                {listPosts.map((post) => (
                  <article key={post.id} className={styles.postCard}>
                    <div
                      className={clsx(
                        styles.postImage,
                        !post.image && styles.postImageFallback,
                      )}
                      style={
                        post.image ? {backgroundImage: `url(${post.image})`} : undefined
                      }
                    />
                    <div className={styles.postBody}>
                      <Heading as="h3" className={styles.postTitle}>
                        {post.title}
                      </Heading>
                      <MetaLine
                        author={post.author}
                        date={post.date}
                        tags={post.tags}
                        readingTime={post.readingTime}
                      />
                      <p className={styles.postExcerpt}>{post.excerpt}</p>
                      <Link className={styles.postLink} to={post.permalink}>
                        Continue Reading →
                      </Link>
                    </div>
                  </article>
                ))}
              </div>
            </div>

            <aside className={styles.sidebar}>
              <div className={styles.sidebarCard}>
                <Heading as="h4" className={styles.sidebarTitle}>
                  Search
                </Heading>
                <div className={styles.searchField}>
                  <input
                    className={styles.searchInput}
                    type="search"
                    placeholder="Type & Hit Enter..."
                  />
                  <button className={styles.searchButton} aria-label="Search">
                    ↵
                  </button>
                </div>
              </div>

              <div className={styles.sidebarCard}>
                <Heading as="h4" className={styles.sidebarTitle}>
                  Hi, I Am John Doe
                </Heading>
                <p className={styles.sidebarText}>
                  A curator of quiet mornings, colorful corners, and thoughtful writing.
                  This space is a home for slow stories and visual notes.
                </p>
                <Link className={styles.sidebarLink} to="/docs/intro">
                  Learn more →
                </Link>
              </div>

              <div className={styles.sidebarCard}>
                <Heading as="h4" className={styles.sidebarTitle}>
                  Recent Posts
                </Heading>
                <ul className={styles.recentList}>
                  {recentPosts.map((post) => (
                    <li key={post.id}>
                      <Link to={post.permalink} className={styles.recentLink}>
                        {post.title}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>

              <div className={styles.sidebarCard}>
                <Heading as="h4" className={styles.sidebarTitle}>
                  Categories
                </Heading>
                <ul className={styles.categoryList}>
                  {categories.map((category) => (
                    <li key={category}>
                      <Link to="/blog" className={styles.categoryLink}>
                        {category}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>

              <div className={styles.sidebarCard}>
                <Heading as="h4" className={styles.sidebarTitle}>
                  Tags
                </Heading>
                <div className={styles.tagList}>
                  {tagList.map((tag) => (
                    <Link key={tag} to="/blog" className={styles.tagChip}>
                      {tag}
                    </Link>
                  ))}
                </div>
              </div>
            </aside>
          </section>
        </div>
      </main>
    </Layout>
  );
}
