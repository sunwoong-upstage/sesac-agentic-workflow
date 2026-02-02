# 뉴스 챗봇 활동 가이드 (강사용)

## 활동 목표

학생들이 뉴스 챗봇 에이전트의 핵심 구성 요소를 직접 구현하면서 LangGraph와 에이전트 개발을 학습합니다.

## 활동 다운로드 및 제출

### 다운로드
1. `STUDENT_ACTIVITIES.md` 파일을 다운로드합니다
2. 본인의 텍스트 에디터에서 엽니다 (VS Code, Cursor 등)

### 제출 방법 (GitHub Gist)
1. [gist.github.com](https://gist.github.com)에 접속
2. GitHub 계정으로 로그인
3. 완성된 마크다운 파일 내용을 붙여넣기
4. "Create secret gist" 또는 "Create public gist" 클릭
5. 생성된 Gist URL을 제출 문서(PDF)에 첨부

## 진행 방법

### 1단계: 마크다운 파일에서 빈칸 채우기
- `STUDENT_ACTIVITIES.md` 다운로드
- 코드 빈칸을 직접 타이핑하여 채우기
- 힌트를 참고하되, 먼저 스스로 생각해보기

### 2단계: 실제 파일에 코드 적용
- `agent/` 폴더의 실제 파일에 코드 복사
- 구문 오류 확인

### 3단계: 실행 및 테스트
```bash
python main.py
```

### 4단계: 제출
- 완성된 마크다운 파일을 GitHub Gist에 업로드
- Gist 링크를 제출 문서에 첨부

## 활동 구조

| 활동 | 대상 파일 | 예상 시간 | 난이도 |
|------|----------|----------|--------|
| 1A-1B | state.py | 30분 | 초급 |
| 2A | prompts.py | 20분 | 초급 |
| 3A-3B | tools.py | 40분 | 중급 |
| 4A | nodes.py | 40분 | 중급 |
| 5A-5B | graph.py | 30분 | 중급 |

## 핵심 개념 매핑

| Practice 노트북 | 핵심 개념 | 뉴스 챗봇 코드 |
|----------------|----------|---------------|
| Practice02 | Plan-and-Solve 패턴 | plan_node, research_node, synthesize_node |
| Practice03 | Tool 정의 (@tool) | search_news_archive, calculate_date_range |
| Practice05 | FAISS 벡터 스토어 | tools.py의 _vector_store 초기화 |
| Practice06 | MemorySaver | graph.py의 checkpointer |
| Practice06 | InMemoryStore | graph.py의 user_store |
| Practice07 | Structured Output | with_structured_output() |
| Practice09 | LLM-as-Judge | evaluate_response_node |

## 평가 기준

- 모든 빈칸이 올바르게 채워졌는지
- `python main.py` 실행 성공 여부
- 코드 이해도 (구두 설명 가능 여부)

## 주의사항

- 복사-붙여넣기 대신 직접 타이핑 권장
- 정답을 먼저 보지 말고 시도해보기
- 오류 발생 시 로그 메시지 확인

## 확장 활동

시간이 남는 학생들을 위한 추가 활동:

| 확장 활동 | 설명 | 난이도 |
|----------|------|--------|
| 감성 분석 도구 | 뉴스 기사의 긍정/부정 분석 | 중급 |
| 카테고리 필터링 | 경제/IT/정치 등 카테고리별 필터 | 중급 |
| 뉴스 요약 노드 | 긴 기사 자동 요약 | 고급 |

## 자주 묻는 질문

### Q: UPSTAGE_API_KEY 오류가 발생합니다
```bash
# .env 파일 확인
cat .env

# 없다면 생성
cp .env.example .env
# 그 후 API 키 입력
```

### Q: FAISS 설치 오류
```bash
pip install faiss-cpu
```

### Q: 의도 분류가 잘 안 됩니다
- 프롬프트의 예시를 더 추가해보세요
- confidence 임계값을 조정해보세요

---

*최종 수정일: 2026-02-02*
