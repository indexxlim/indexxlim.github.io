# TODO 리스트 점검 보고서

## 프로젝트 개요

**목표**: [Logbook Hugo 테마](https://logbook-hugo.vercel.app/) 스타일의 미니멀 editorial/magazine 블로그를 Docusaurus로 구현

**현재 상태**: 22개 중 12개 작업 완료 (54.5%), 현재 반응형 디자인 작업 진행 중

---

## ✅ 완료된 작업 (12개)

### 디자인 기반 구축
1. ✅ **색상 스킴 업데이트** - 중립적 미니멀 팔레트 적용
   - Primary: `#1a1a1a` (다크), Accent: `#c46a3b` (웜 오렌지)
   - Background: `#fcfaf6` (오프화이트)
   - [src/css/custom.css](src/css/custom.css) 구현 완료

2. ✅ **타이포그래피 계층** - 깔끔한 sans-serif 적용
   - Heading: `Playfair Display` (serif, decorative)
   - Body: `Source Sans 3` (sans-serif)
   - 명확한 hierarchy 구현 완료

3. ✅ **Navbar 커스터마이징** - Blog-first 레이아웃
   - IndexxLim 로고 및 브랜딩
   - Blog/Docs 네비게이션, GitHub 링크
   - [docusaurus.config.js](docusaurus.config.js) 설정 완료

4. ✅ **Footer 재디자인** - 멀티컬럼 레이아웃
   - Content / Social / Topics 3단 구성
   - 참조 사이트와 동일한 구조

### 블로그 포스트 레이아웃
5. ✅ **커스텀 블로그 카드 레이아웃** - 강화된 스타일링
   - [src/theme/BlogPostItem/index.js](src/theme/BlogPostItem/index.js)
   - 카드 뷰와 전체 article 뷰 모두 지원

6. ✅ **포스트 메타데이터 표시** - author, date, tags with pill styling
   - 태그 pill 스타일 (hover 효과 포함)
   - 작성자, 날짜 정보 표시

7. ✅ **카드 기반 블로그 리스팅** - hover effects
   - 0-4px elevation on hover
   - subtle shadow transitions

8. ✅ **블로그 페이지 너비 수정** - 1400px
   - main/docs 페이지와 일관성 유지

9. ✅ **Featured 이미지 및 설명 추가**
   - frontMatter에서 `image` 또는 `cover_image` 지원
   - 최대 400px height로 카드에 표시

### 사이드바 및 컴포넌트
10. ✅ **사이드바 구현** - search, author bio, categories, tags, latest posts
    - [src/theme/BlogSidebar/index.js](src/theme/BlogSidebar/index.js)
    - 모든 핵심 섹션 완료

11. ✅ **Author bio 섹션** - 블로그 포스트용
    - [src/components/AuthorBio/index.js](src/components/AuthorBio/index.js)
    - 프로필 이미지, 소셜 링크 포함

12. ✅ **소셜 미디어 공유 버튼**
    - [src/components/SocialShare/index.js](src/components/SocialShare/index.js)
    - Twitter, LinkedIn, Facebook, Reddit 지원

---

## ⏳ 진행 중 (1개)

### 현재 작업: Step 1 - Responsive Design
**README 명시**: "Current focus: Step 1 (responsive design)."

- [ ] **모바일/태블릿 반응형 디자인 테스트**
  - 사용자가 선택한 라인: [README.md:23-24](README.md#L23-L24)
  - 필요한 점검:
    - 블로그 카드 레이아웃 (모바일에서 1-column으로 전환)
    - 사이드바 동작 (모바일에서 collapse/toggle)
    - 네비게이션 메뉴 (hamburger menu)
    - Featured 이미지 크기 조정
    - 타이포그래피 스케일링
    - 간격 및 패딩 조정

---

## 📋 남은 작업 (9개)

### Step-by-step Plan (README 순서)

#### Step 1: 반응형 디자인 (현재)
- [ ] Responsive design audit + fixes (mobile/tablet layout, spacing, sidebar behavior)

#### Step 2: 애니메이션 개선
- [ ] Blog card hover/animation polish
  - 현재 hover effects 있지만 더 polish 필요

#### Step 3: 읽기 시간 추정
- [ ] Reading time estimates on blog posts
  - Docusaurus는 기본 제공하지 않음 → 커스텀 플러그인 필요

#### Step 4: 목차 (TOC)
- [ ] Table of contents for long blog posts
  - Docusaurus는 기본 TOC 제공하지만 커스터마이징 필요할 수 있음

#### Step 5: 다크모드
- [ ] Dark mode toggle with smooth transitions
  - 현재 다크모드 CSS 변수는 정의되어 있지만 toggle UI 미구현

#### Step 6: 이미지 최적화
- [ ] Featured image loading/sizing optimization
  - Lazy loading, responsive images, webp 형식 등

#### Step 7: Breadcrumb
- [ ] Breadcrumb navigation
  - Docusaurus는 docs에 기본 breadcrumbs 있지만 blog에는 없음

#### Step 8: 관련 포스트
- [ ] Related posts section
  - 태그 기반 또는 카테고리 기반 추천 로직 필요

#### Step 9: Newsletter
- [ ] Newsletter subscription form
  - 외부 서비스 통합 (Mailchimp, ConvertKit 등)

---

## 🎯 우선순위 제안

### 높음 (핵심 사용자 경험)
1. **Responsive design** (현재 진행 중) - 모바일 사용자 필수
2. **Dark mode toggle** - 현대 웹사이트 표준 기능
3. **Reading time estimates** - 참조 사이트 핵심 기능

### 중간 (UX 개선)
4. **Table of contents** - 긴 글 읽기 편의성
5. **Blog card hover polish** - 시각적 polish
6. **Featured image optimization** - 성능 개선

### 낮음 (추가 기능)
7. **Breadcrumb navigation** - 선택적 기능
8. **Related posts** - 콘텐츠가 많아지면 유용
9. **Newsletter** - 독자 확보 후 구현

---

## 📊 참조 사이트 대비 구현 현황

### Logbook Hugo 주요 특징 vs 현재 구현

| 기능 | Logbook Hugo | 현재 구현 | 상태 |
|------|-------------|----------|------|
| Editorial/magazine style | ✓ | ✓ | ✅ 완료 |
| Minimalist color palette | ✓ | ✓ | ✅ 완료 |
| Large featured images | ✓ | ✓ | ✅ 완료 |
| Multi-column layout | ✓ | ✓ | ✅ 완료 |
| Rich sidebar | ✓ | ✓ | ✅ 완료 |
| Typography hierarchy | ✓ | ✓ | ✅ 완료 |
| Multi-column footer | ✓ | ✓ | ✅ 완료 |
| Responsive design | ✓ | ⏳ | 🔄 진행 중 |
| Dark mode toggle | ✓ | ✗ | ⏸️ 대기 |
| Reading time | ✓ | ✗ | ⏸️ 대기 |
| Newsletter form | ✓ | ✗ | ⏸️ 대기 |

---

## 🔍 반응형 디자인 구현 계획

### 현재 상태 분석

**이미 구현된 반응형 기능**:
- BlogListPage: 1200px에서 사이드바가 상단으로 이동, 768px에서 single column
- BlogPostItem: 768px에서 padding 및 font-size 조정
- BlogSidebar: 768px에서 작은 아바타, 패딩 감소
- Typography: 996px, 768px에서 font scaling

**기존 breakpoints**: 1200px, 996px, 768px, 480px

**개선 필요 사항**:
- 태블릿 최적화 (768px-996px) 부족
- 모바일에서 max-width 제약 없음
- Featured 이미지 모바일 aspect ratio 최적화 필요
- 터치 타겟 44x44px 검증 필요
- 사이드바가 모바일에서 컨텐츠 위에 표시 (order: -1) - UX 개선 가능

---

## 📋 구현 우선순위

### Phase 1: Critical (필수)
1. **custom.css** - 컨테이너 overflow 방지, 패딩 조정
2. **BlogListPage** - 모바일 레이아웃 개선
3. **SocialShare** - 터치 타겟 44x44px 확보
4. **테스트**: iPhone SE (375px), iPad (768px)

### Phase 2: Important (중요)
5. **BlogListPage/BlogPostItem** - 태블릿 breakpoint 추가 (768-996px)
6. **BlogSidebar** - 태블릿 너비 최적화
7. **Featured 이미지** - 모바일 aspect ratio 조정
8. **테스트**: iPad (768px, 1024px)

### Phase 3: Nice-to-have (선택)
9. **Extra-small mobile** - <480px breakpoint 추가
10. **Spacing 세부 조정** - 모든 breakpoint에서 gap/padding 최적화
11. **테스트**: iPhone SE 1st gen (320px)

---

## 🎯 Breakpoint 전략

### 화면 크기별 레이아웃

**Desktop Large (>1200px)**: 기본 스타일
- 1400px max-width
- 2-column (main + 300px sidebar)
- Featured 이미지 400px max-height
- Sidebar sticky positioning

**Desktop (996px-1200px)**: 약간 축소
- 2-column (main + 280px sidebar)
- Gap 1.75rem

**Tablet (768px-996px)**: 중간 크기 최적화 ⭐ 추가 필요
- 2-column (main + 250px sidebar) 또는 single column
- Gap 1.5rem
- Featured 이미지 320px max-height, aspect-ratio 55%
- Sidebar static (not sticky)

**Mobile (480px-768px)**: 모바일 레이아웃
- Single column
- Sidebar 컨텐츠 아래로 이동 (order 변경)
- 1rem 패딩
- Featured 이미지 280px max-height, aspect-ratio 60%
- Gap 1.5rem

**Mobile Small (<480px)**: 최소 간격
- 0.75rem 패딩
- Featured 이미지 aspect-ratio 65%
- Gap 1rem
- 터치 타겟 44x44px 확인

### Media Query 구조
```css
/* Desktop Large - default (>1200px) */

/* Desktop (996px-1200px) */
@media (max-width: 1200px) and (min-width: 997px) { }

/* Tablet (768px-996px) ⭐ 추가 */
@media (max-width: 996px) and (min-width: 769px) { }

/* Mobile (480px-768px) */
@media (max-width: 768px) and (min-width: 481px) { }

/* Mobile Small (<480px) */
@media (max-width: 480px) { }
```

---

## 📂 수정할 파일 및 변경 내용

### 1. [src/css/custom.css](src/css/custom.css) - 최우선

**변경 사항**:
- ✅ Tablet (768-996px): container 100% width, padding 0 1.5rem, main padding 1.5rem 0 3rem
- ✅ Mobile (<768px): container padding 0 1rem, main padding 1rem 0 2rem, featured image 280px
- ✅ Extra-small (<480px): container padding 0 0.75rem, featured image aspect-ratio 65%
- ✅ Overflow 방지: max-width 100%

### 2. [src/theme/BlogListPage/styles.module.css](src/theme/BlogListPage/styles.module.css) - 핵심

**변경 사항**:
- ✅ Desktop (996-1200px): sidebar 280px, gap 1.75rem
- ✅ Tablet (768-996px): sidebar 250px, gap 1.5rem, grid minmax(280px, 1fr)
- ✅ Mobile (<768px): padding 1rem, gap 1.5rem, sidebar 컨텐츠 아래로 (order 변경)
- ✅ Extra-small (<480px): padding 0.75rem, gap 1rem

**사이드바 위치 결정**: 모바일에서 sidebar를 컨텐츠 **아래**로 이동 (content-first)

### 3. [src/theme/BlogPostItem/styles.module.css](src/theme/BlogPostItem/styles.module.css) - 중요

**변경 사항**:
- ✅ Tablet (768-996px): cardContent padding 1.5rem, imageWrapper padding-top 55%
- ✅ Mobile (<768px): imageWrapper padding-top 60%, h2 1.2rem
- ✅ Extra-small (<480px): cardContent padding 0.875rem, h2 1.125rem, metadata 0.75rem

### 4. [src/theme/BlogSidebar/styles.module.css](src/theme/BlogSidebar/styles.module.css) - 중요

**변경 사항**:
- ✅ Desktop (996-1200px): section padding 1.25rem
- ✅ Tablet (768-996px): 현재 레이아웃 유지
- ✅ Mobile (<768px): section padding 1rem, tagCloud 2-column grid, avatar 56px
- ✅ Extra-small (<480px): section padding 1rem, avatar 48px

### 5. [src/components/SocialShare/styles.module.css](src/components/SocialShare/styles.module.css) - 터치 타겟

**변경 사항**:
- ✅ Mobile (<768px): shareButton padding 0.625rem 1rem (min-height 44px 확보)
- ✅ Extra-small (<480px): 필요시 full-width 버튼

---

## ✅ 테스트 전략

### 수동 테스트 화면 크기
1. Desktop Large: 1920x1080, 1440x900
2. Desktop: 1200x800, 1024x768
3. Tablet Portrait: 768x1024 (iPad)
4. Tablet Landscape: 1024x768 (iPad)
5. Mobile Large: 414x896 (iPhone 11 Pro Max)
6. Mobile Medium: 375x667 (iPhone SE)
7. Mobile Small: 320x568 (iPhone SE 1st gen)

### Chrome DevTools 테스트
```bash
# 로컬 서버 시작
yarn start

# DevTools에서 테스트할 breakpoint
- 320px (extra-small)
- 480px (small mobile)
- 768px (mobile/tablet 경계)
- 996px (tablet/desktop 경계)
- 1200px (desktop large)
- 1400px (max-width)
```

### 검증 체크리스트
- [ ] 모든 화면 크기에서 horizontal scrollbar 없음
- [ ] 사이드바가 올바르게 동작 (desktop: sticky, tablet: static, mobile: 하단)
- [ ] 블로그 카드가 모바일에서 single column
- [ ] Featured 이미지 aspect ratio 유지
- [ ] 터치 타겟 최소 44x44px (모바일)
- [ ] 타이포그래피 가독성 (최소 14px)
- [ ] 적절한 padding (텍스트가 edge에 붙지 않음)

---

## ⚠️ 주의사항

### Docusaurus 기본 스타일 충돌
- Infima CSS 프레임워크 기본 breakpoint: 996px
- CSS 모듈로 격리되어 있어 충돌 가능성 낮음
- Production build에서 최종 테스트 필요

### Swizzled 컴포넌트
- BlogListPage, BlogPostItem, BlogSidebar는 안전하게 swizzle됨
- Docusaurus 업데이트 시 호환성 확인 필요

### 사이드바 모바일 동작
- **현재**: 사이드바가 컨텐츠 위 (order: -1)
- **변경**: 사이드바를 컨텐츠 아래로 이동 (content-first)
- **향후 개선**: Collapsible sections (JavaScript 필요)

### 이미지 Aspect Ratio
- 현재: padding-top 52% (고정)
- 변경: Breakpoint별로 다른 aspect ratio (55%, 60%, 65%)
- 주의: 실제 이미지로 테스트하여 중요 콘텐츠 cropping 확인

---

## 🔧 구현 순서

### Step 1: custom.css 수정 (30분)
- Container max-width, padding 조정
- Tablet, Mobile, Extra-small breakpoint 추가
- Featured 이미지 max-height 조정
- Overflow 방지

### Step 2: BlogListPage 수정 (45분)
- Tablet breakpoint 추가 (sidebar 250px)
- Mobile 사이드바 순서 변경 (컨텐츠 우선)
- Padding, gap 조정

### Step 3: BlogPostItem 수정 (45분)
- Tablet breakpoint 추가
- Featured 이미지 aspect ratio 모바일 최적화
- Typography, padding 조정

### Step 4: BlogSidebar 수정 (30분)
- Tablet breakpoint 추가
- Mobile tagCloud 2-column grid
- Avatar 크기 조정

### Step 5: SocialShare 수정 (15분)
- Mobile 터치 타겟 44x44px 확보
- Button padding 증가

### Step 6: 테스트 및 수정 (1-2시간)
- Chrome DevTools로 모든 breakpoint 테스트
- 실제 디바이스 테스트 (가능하면)
- 발견된 이슈 수정
- Lighthouse 모바일 audit

**예상 소요 시간**: 4-5시간

---

## 📝 최종 권장사항

1. **Phase 1부터 순차적으로 구현**: 각 Phase 완료 후 테스트하여 점진적으로 개선
2. **실제 블로그 포스트로 테스트**: 24개의 기존 포스트로 이미지, 콘텐츠 레이아웃 검증
3. **모바일 우선**: Mobile/Tablet 사용자가 많으므로 우선순위 높음
4. **Production build 테스트**: Dev 서버와 production build에서 스타일이 다를 수 있음
5. **README 업데이트**: 구현 완료 후 반응형 디자인 체크박스 체크

이 계획을 실행하면 Logbook Hugo 테마의 반응형 디자인을 성공적으로 Docusaurus로 구현할 수 있습니다.

---

## 📝 결론

**진행률**: 54.5% 완료 (12/22)

**현재 상태**: 핵심 디자인과 레이아웃은 완성되었으며, Logbook Hugo 테마의 editorial/magazine 스타일을 성공적으로 구현했습니다. 24개의 블로그 포스트가 있으며, featured 이미지, 태그, 카테고리, 사이드바 등 주요 기능이 모두 작동합니다.

**다음 단계**: 반응형 디자인 audit 및 수정이 최우선 과제입니다. 모바일/태블릿 사용자 경험을 개선한 후, dark mode toggle과 reading time estimates를 순차적으로 구현하는 것을 권장합니다.

**장점**:
- 깔끔한 코드 구조 (Docusaurus best practices 준수)
- 재사용 가능한 컴포넌트 설계
- 명확한 README-driven 개발 프로세스

**개선 필요**:
- 반응형 디자인 (현재 진행 중)
- 다크모드 UI toggle
- 성능 최적화 (이미지 로딩)
