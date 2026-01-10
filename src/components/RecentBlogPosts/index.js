import React from 'react';
import clsx from 'clsx';
import {BlogPostProvider} from '@docusaurus/plugin-content-blog/client';
import {HtmlClassNameProvider} from '@docusaurus/theme-common';
import BlogListPaginator from '@theme/BlogListPaginator';
import BlogListPage from '@theme/BlogListPage';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

// 최근 블로그 포스트를 표시하는 컴포넌트 (플레이스홀더)
export default function RecentBlogPosts() {
  return (
    <section className={styles.blogSection}>
      <div className="container">
        <Heading as="h2" className={styles.sectionTitle}>
          Recent Blog Posts
        </Heading>
        <p className={styles.blogPlaceholder}>
          Blog posts will appear here. Visit <Link to="/blog">the blog page</Link> to see all posts.
        </p>
      </div>
    </section>
  );
}
