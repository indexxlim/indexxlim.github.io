import React from 'react';
import clsx from 'clsx';
import {useBlogPost} from '@docusaurus/plugin-content-blog/client';
import BlogPostItemHeaderTitle from '@theme/BlogPostItem/Header/Title';
import BlogPostItemHeaderInfo from '@theme/BlogPostItem/Header/Info';
import BlogPostItemHeaderAuthors from '@theme/BlogPostItem/Header/Authors';
import BlogPostItemContent from '@theme/BlogPostItem/Content';
import BlogPostItemFooter from '@theme/BlogPostItem/Footer';
import AuthorBio from '@site/src/components/AuthorBio';
import SocialShare from '@site/src/components/SocialShare';
import styles from './styles.module.css';

export default function BlogPostItem({children, className}) {
  const {metadata, isBlogPostPage} = useBlogPost();
  const {
    permalink,
    title,
    date,
    formattedDate,
    readingTime,
    tags,
    authors,
    frontMatter,
  } = metadata;

  const image = frontMatter.image || frontMatter.cover_image;

  // Card layout for blog list page
  if (!isBlogPostPage) {
    return (
      <article className={clsx(styles.blogPostCard, className)}>
        {image && (
          <a href={permalink} className={styles.imageLink}>
            <div className={styles.imageWrapper}>
              <img src={image} alt={title} className={styles.featuredImage} />
            </div>
          </a>
        )}
        <div className={styles.cardContent}>
          <header>
            <BlogPostItemHeaderTitle />
            <div className={styles.metadata}>
              <BlogPostItemHeaderInfo />
              <BlogPostItemHeaderAuthors />
            </div>
          </header>
          <BlogPostItemContent>{children}</BlogPostItemContent>
          <BlogPostItemFooter />
        </div>
      </article>
    );
  }

  // Full page layout for individual blog post
  const baseUrl = typeof window !== 'undefined' ? window.location.origin : 'https://indexxlim.github.io';
  const fullUrl = `${baseUrl}${permalink}`;

  return (
    <article className={clsx('margin-bottom--xl', className)}>
      <BlogPostItemHeaderTitle />
      <BlogPostItemHeaderInfo />
      <BlogPostItemHeaderAuthors />
      <BlogPostItemContent>{children}</BlogPostItemContent>
      <BlogPostItemFooter />
      <SocialShare url={fullUrl} title={title} />
      <AuthorBio author={authors?.[0]} />
    </article>
  );
}
