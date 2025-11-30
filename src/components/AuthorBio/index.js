import React from 'react';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

export default function AuthorBio({author}) {
  const defaultAuthor = {
    name: 'IndexxLim',
    title: 'ML/NLP Researcher',
    url: 'https://github.com/indexxlim',
    imageURL: '/img/profile.jpg',
    description: 'Passionate about Machine Learning and Natural Language Processing. Sharing insights on ML research, paper reviews, and practical implementations.',
    socials: {
      github: 'https://github.com/indexxlim',
      linkedin: 'https://www.linkedin.com/in/indexxlim',
    }
  };

  const authorData = author || defaultAuthor;

  return (
    <div className={styles.authorBio}>
      <div className={styles.authorHeader}>
        <img
          src={authorData.imageURL}
          alt={authorData.name}
          className={styles.authorImage}
          onError={(e) => {
            e.target.src = '/img/logo.svg';
          }}
        />
        <div className={styles.authorInfo}>
          <h3 className={styles.authorName}>{authorData.name}</h3>
          {authorData.title && (
            <p className={styles.authorTitle}>{authorData.title}</p>
          )}
        </div>
      </div>
      {authorData.description && (
        <p className={styles.authorDescription}>{authorData.description}</p>
      )}
      {authorData.socials && (
        <div className={styles.authorSocials}>
          {authorData.socials.github && (
            <a
              href={authorData.socials.github}
              target="_blank"
              rel="noopener noreferrer"
              className={styles.socialLink}>
              GitHub
            </a>
          )}
          {authorData.socials.linkedin && (
            <a
              href={authorData.socials.linkedin}
              target="_blank"
              rel="noopener noreferrer"
              className={styles.socialLink}>
              LinkedIn
            </a>
          )}
          {authorData.socials.twitter && (
            <a
              href={authorData.socials.twitter}
              target="_blank"
              rel="noopener noreferrer"
              className={styles.socialLink}>
              Twitter
            </a>
          )}
        </div>
      )}
    </div>
  );
}
