import {useEffect, useMemo, useState} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import blogPostListProp from '@generated/docusaurus-plugin-content-blog/default/blog-post-list-prop-default.json';
import styles from './index.module.css';

// 슬라이더 자동 회전 간격 (밀리초)
const FALLBACK_AUTOROTATE_MS = 6500;

// 날짜 문자열을 읽기 쉬운 형식으로 변환 (예: "January 10, 2026")
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

// 읽기 시간을 분 단위로 포맷팅 (최소 1분)
const formatReadingTime = (readingTime) => {
  if (!readingTime) {
    return '';
  }
  const minutes = Math.max(1, Math.round(readingTime));
  return `${minutes} min read`;
};

// 마크다운 문법을 제거하고 순수 텍스트만 추출
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

// 텍스트에서 요약문 추출 (최대 길이 제한하고 말줄임표 추가)
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

// 블로그 메타데이터 JSON 파일들을 동적으로 로드하기 위한 컨텍스트
let blogMetadataContext;
try {
  blogMetadataContext = require.context(
    '@generated/docusaurus-plugin-content-blog/default',
    false,
    /site-blog-.*\.json$/,
  );
} catch (error) {
  blogMetadataContext = null;
}

// 작성자, 날짜, 태그, 읽기시간을 한 줄로 표시하는 메타 정보 컴포넌트
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

// 메인 홈페이지 컴포넌트 - LogBook 스타일의 저널형 레이아웃
export default function Home() {
  // 블로그 포스트 목록 가져오기
  const blogPosts = blogPostListProp?.items ?? [];
  // permalink를 키로 하는 메타데이터 맵 생성 (성능 최적화를 위해 useMemo 사용)
  const metadataByPermalink = useMemo(() => {
    if (!blogMetadataContext) {
      return {};
    }
    const entries = {};
    blogMetadataContext.keys().forEach((key) => {
      const entry = blogMetadataContext(key);
      if (entry?.permalink) {
        entries[entry.permalink] = entry;
      }
    });
    return entries;
  }, []);

  // 블로그 포스트를 표준화된 형식으로 변환 (이미지, 태그, 설명 등 추가)
  const normalizedPosts = useMemo(
    () =>
      blogPosts.map((post) => {
        const metadata = metadataByPermalink[post.permalink];
        const frontMatter = metadata?.frontMatter ?? {};
        const tags = metadata?.tags?.map((tag) => tag.label) ?? [];
        const description = metadata?.description || frontMatter.description || '';

        return {
          id: post.permalink,
          title: post.title ?? 'Untitled',
          permalink: post.permalink ?? '/blog',
          date: post.date,
          author: metadata?.authors?.[0]?.name ?? '',
          tags,
          readingTime: metadata?.readingTime,
          image: frontMatter.image || frontMatter.cover_image || '',
          excerpt: getExcerpt(description),
        };
      }),
    [blogPosts, metadataByPermalink],
  );

  // 히어로 슬라이더에 표시할 추천 포스트 (최대 3개)
  const featuredPosts = normalizedPosts.slice(0, 3);
  // 포스트 그리드에 표시할 목록 (첫 번째 제외하고 최대 5개)
  const listPosts = normalizedPosts.slice(featuredPosts.length ? 1 : 0, 5);
  // 사이드바 최근 포스트 목록 (최대 3개)
  const recentPosts = normalizedPosts.slice(0, 3);
  // 모든 태그를 중복 없이 추출
  const uniqueTags = Array.from(
    new Set(normalizedPosts.flatMap((post) => post.tags)),
  );
  // 사이드바 카테고리 목록 (최대 5개)
  const categories = uniqueTags.slice(0, 5);
  // 사이드바 태그 칩 목록 (최대 8개)
  const tagList = uniqueTags.slice(0, 8);

  // 현재 활성화된 슬라이드 인덱스 상태
  const [activeIndex, setActiveIndex] = useState(0);

  // 슬라이드 자동 회전 효과 (접근성 고려: prefers-reduced-motion 체크)
  useEffect(() => {
    // 슬라이드가 1개 이하면 자동 회전 불필요
    if (featuredPosts.length <= 1) {
      return undefined;
    }
    // 서버 사이드 렌더링에서는 실행 안함
    if (typeof window === 'undefined') {
      return undefined;
    }
    // 사용자가 애니메이션 감소를 선호하면 자동 회전 비활성화
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (mediaQuery.matches) {
      return undefined;
    }
    // 일정 시간마다 다음 슬라이드로 자동 전환
    const id = window.setInterval(() => {
      setActiveIndex((prev) => (prev + 1) % featuredPosts.length);
    }, FALLBACK_AUTOROTATE_MS);
    return () => window.clearInterval(id);
  }, [featuredPosts.length]);

  // 활성 인덱스가 범위를 벗어나면 0으로 리셋
  useEffect(() => {
    if (activeIndex > featuredPosts.length - 1) {
      setActiveIndex(0);
    }
  }, [activeIndex, featuredPosts.length]);

  // 현재 활성화된 포스트와 슬라이드 가능 여부
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
                    {activePost?.excerpt ? (
                      <p className={styles.heroExcerpt}>{activePost.excerpt}</p>
                    ) : null}
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
                      {post.excerpt ? (
                        <p className={styles.postExcerpt}>{post.excerpt}</p>
                      ) : null}
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
