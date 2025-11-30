import React from 'react';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

function SearchBox() {
  return (
    <div className={styles.sidebarSection}>
      <h3 className={styles.sidebarTitle}>Search</h3>
      <div className={styles.searchBox}>
        <input
          type="text"
          placeholder="Search posts..."
          className={styles.searchInput}
          onFocus={(e) => {
            // Trigger Docusaurus search if available
            const searchButton = document.querySelector('.DocSearch-Button');
            if (searchButton) {
              searchButton.click();
            }
          }}
        />
      </div>
    </div>
  );
}

function AuthorBio() {
  return (
    <div className={styles.sidebarSection}>
      <h3 className={styles.sidebarTitle}>About</h3>
      <div className={styles.authorBio}>
        <div className={styles.authorAvatar}>
          <img
            src="/img/profile.jpg"
            alt="IndexxLim"
            onError={(e) => {
              e.target.src = '/img/logo.svg';
            }}
          />
        </div>
        <h4 className={styles.authorName}>IndexxLim</h4>
        <p className={styles.authorDescription}>
          ML/NLP Researcher sharing insights on machine learning, natural language processing, and AI research.
        </p>
        <div className={styles.authorLinks}>
          <a href="https://github.com/indexxlim" target="_blank" rel="noopener noreferrer">
            GitHub
          </a>
          <a href="https://www.linkedin.com/in/indexxlim" target="_blank" rel="noopener noreferrer">
            LinkedIn
          </a>
        </div>
      </div>
    </div>
  );
}

function Categories() {
  const categories = [
    { name: 'NLP', slug: 'nlp' },
    { name: 'Transformer', slug: 'transformer' },
    { name: 'Deep Learning', slug: 'deep-learning' },
    { name: 'ASR', slug: 'asr' },
    { name: 'LLM', slug: 'llm' },
  ];

  return (
    <div className={styles.sidebarSection}>
      <h3 className={styles.sidebarTitle}>Categories</h3>
      <ul className={styles.categoryList}>
        {categories.map((category) => (
          <li key={category.slug}>
            <Link to={`/blog/tags/${category.slug}`}>
              {category.name}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}

function PopularTags() {
  const tags = [
    { name: 'NLP', slug: 'nlp' },
    { name: 'Transformer', slug: 'transformer' },
    { name: 'Deep Learning', slug: 'deep-learning' },
    { name: 'LLM', slug: 'llm' },
    { name: 'BERT', slug: 'bert' },
    { name: 'Attention', slug: 'attention' },
    { name: 'ASR', slug: 'asr' },
    { name: 'Pre-training', slug: 'pre-training' },
  ];

  return (
    <div className={styles.sidebarSection}>
      <h3 className={styles.sidebarTitle}>Popular Tags</h3>
      <div className={styles.tagCloud}>
        {tags.map((tag) => (
          <Link
            key={tag.slug}
            to={`/blog/tags/${tag.slug}`}
            className={styles.tag}>
            {tag.name}
          </Link>
        ))}
      </div>
    </div>
  );
}

function LatestPosts({items = []}) {
  // Get latest 5 posts and filter out invalid items
  const latestPosts = items
    .filter(item => item && item.content && item.content.metadata)
    .slice(0, 5);

  if (latestPosts.length === 0) {
    return null;
  }

  return (
    <div className={styles.sidebarSection}>
      <h3 className={styles.sidebarTitle}>Latest Posts</h3>
      <ul className={styles.postList}>
        {latestPosts.map((item) => {
          const {content} = item;
          const {metadata} = content;
          const {title, permalink, formattedDate} = metadata;

          return (
            <li key={permalink}>
              <Link to={permalink} className={styles.postLink}>
                <span className={styles.postTitle}>{title}</span>
                <span className={styles.postDate}>{formattedDate}</span>
              </Link>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export default function BlogSidebar({sidebar}) {
  // Try to get blog posts from sidebar items
  const items = sidebar?.items || [];

  return (
    <div className={styles.sidebar}>
      <SearchBox />
      <AuthorBio />
      <Categories />
      <PopularTags />
      {items.length > 0 && <LatestPosts items={items} />}
    </div>
  );
}
