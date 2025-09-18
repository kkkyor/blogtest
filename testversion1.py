import streamlit as st
import time
import json
import os
import pyperclip
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
# LangChain 연동을 위한 라이브러리 임포트
from langchain_naver import ChatClovaX

# --- 1. 네이버 CLOVA API 연동 함수 (LangChain 적용) ---

def generate_blog_post_from_review(clova_studio_api_key: str, review_text: str) -> tuple[str, str, list]:
    """
    langchain-naver 라이브러리를 사용하여 CLOVA X API를 호출하고,
    입력된 계약 리뷰를 바탕으로 블로그 게시글(제목, 본문, 태그)을 생성합니다.

    Args:
        clova_studio_api_key (str): 네이버 CLOVA Studio API 키.
        review_text (str): 사용자가 입력한 계약 리뷰 텍스트.

    Returns:
        tuple[str, str, list]: 생성된 블로그의 (제목, 본문, 태그 리스트).
                               오류 발생 시, (에러 메시지, 상세 내용, 빈 리스트)를 반환합니다.
    """
    # LangChain은 환경 변수를 통해 API 키를 읽는 것을 권장합니다.
    os.environ["CLOVASTUDIO_API_KEY"] = clova_studio_api_key

    try:
        # 1. LangChain의 ChatClovaX 모델 초기화
        chat = ChatClovaX(
            model="HCX-003",
            max_tokens=1500,
            temperature=0.7,
            top_p=0.8,
            repeat_penalty=5.0
        )

        # 2. AI에게 역할을 부여하고, 원하는 결과물의 형식을 JSON으로 명확하게 지정하는 프롬프트
        prompt = f"""
당신은 기업의 성공적인 고객 사례를 바탕으로, 잠재 고객의 마음을 사로잡는 스토리텔러이자 전문 마케터입니다.

[콘텐츠 작성 방향 및 구조]
- 단순한 정보 나열이 아닌, 고객이 겪는 '문제'에서 '해결'에 이르는 한 편의 '성공 스토리'를 서사적으로 구성해주세요.
- 독자가 전문가와 편안하게 대화하는 것처럼 느끼도록, **친근하고 부드러운 어조**로 작성해주세요.
- 딱딱하거나 어려운 전문 용어보다는 쉬운 단어를 사용해주세요.
- **다음 단어들의 사용은 피해주세요: '획기적인', '솔루션', '시너지'**
- 아래 4단계 구조에 따라 글을 작성해주세요.
    1. 도입 (공감대 형성): 고객이 겪었던 어려움을 생생하게 묘사하며 독자의 공감을 유도합니다.
    2. 전개 (문제 해결 여정): 우리가 어떻게 고객의 문제를 이해하고 맞춤형 방안을 제시했는지 과정을 설명합니다.
    3. 절정 (핵심 가치 전달): 우리 제품/서비스가 어떤 핵심적인 역할을 했는지 명확하게 보여줍니다.
    4. 결말 (긍정적 미래 제시 및 행동 유도): 계약 리뷰의 '기대 효과'를 바탕으로 고객의 밝은 미래를 그려주고, 비슷한 고민을 하는 다른 잠재 고객들에게 문의나 추가 정보 확인을 유도하는 문장(Call to Action)으로 마무리합니다.

아래에 제공되는 영업사원의 계약 리뷰 노트를 바탕으로, 다음 JSON 형식에 맞춰 **반드시 한 줄짜리(single-line) 문자열로만** 응답해주세요.
JSON 구조 자체에 줄바꿈을 포함해서는 안 됩니다.
**중요: "content" 값 내부에 포함되는 모든 줄바꿈과 특수문자는 반드시 유효한 JSON 형식에 맞게 이스케이프(escape) 처리해야 합니다. (예: 줄바꿈은 \\n 으로 표현)**

- "title": 블로그 제목 (검색에 유리하도록 고객사와 핵심 해결 방안을 포함)
- "content": 블로그 본문 (Markdown 형식, 소제목 사용)
- "tags": 추천 해시태그 5개 (리스트 형식)

---
[계약 리뷰 노트]
{review_text}
---
"""
        # 3. 모델 호출 (invoke 사용)
        ai_msg = chat.invoke(prompt)

        # 4. AI 응답에서 순수한 JSON 부분만 추출 (가끔 응답 앞뒤에 불필요한 텍스트가 붙는 경우 방지)
        response_text = ai_msg.content
        json_start_index = response_text.find('{')
        json_end_index = response_text.rfind('}') + 1

        if json_start_index != -1 and json_end_index != 0:
            json_str = response_text[json_start_index:json_end_index]
            result_json = json.loads(json_str)
        else:
            raise ValueError("AI 응답에서 유효한 JSON 객체를 찾을 수 없습니다.")

        return result_json.get("title", ""), result_json.get("content", ""), result_json.get("tags", [])

    except Exception as e:
        st.error(f"CLOVA API 호출 중 오류가 발생했습니다: {e}")
        return "API 호출 실패", f"오류가 발생했습니다. API 키와 입력 내용을 확인해주세요. \n\n상세 오류: {e}", []


# --- 2. Selenium 블로그 포스팅 함수 ---

def post_to_naver_blog(title: str, content: str, tags: list):
    """
    Selenium을 이용해 네이버 블로그에 자동으로 글을 발행합니다.
    사용자가 직접 한 번만 로그인하면, 지정된 프로필 폴더에 세션이 유지됩니다.

    Args:
        title (str): 블로그 게시글 제목.
        content (str): 블로그 게시글 본문 (Markdown).
        tags (list): 해시태그 리스트.

    Returns:
        bool: 포스팅 성공 여부.
    """
    driver = None # 드라이버 변수 초기화
    try:
        # --- 드라이버 설정 (로그인 세션 유지를 위해 프로필 경로 지정) ---
        st.info("Chrome 브라우저를 설정하고 실행합니다...")
        profile_path = os.path.join(os.getcwd(), "naver_blog_profile")
        options = webdriver.ChromeOptions()
        options.add_argument(f"user-data-dir={profile_path}")
        # 안정성 향상을 위한 옵션 (특히 서버 환경에서 유용)
        # options.add_argument("--headless")  # 백그라운드 실행. 최초 로그인 시에는 주석 처리 필요.
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        wait = WebDriverWait(driver, 20)

        # --- 네이버 블로그 글쓰기 페이지 접속 ---
        # 특정 카테고리에 글을 작성하도록 URL 지정
        write_url = "https://blog.naver.com/kkkyor?Redirect=Write&categoryNo=1"
        driver.get(write_url)
        time.sleep(3)

        # --- 로그인 상태 확인 및 안내 ---
        if "login" in driver.current_url.lower():
            st.warning("⚠️ **최초 1회 수동 로그인이 필요합니다.**")
            st.info(
                "1. 지금 열린 Chrome 브라우저에서 네이버 로그인을 진행해주세요.\n"
                "2. **'로그인 상태 유지'**에 반드시 체크하세요.\n"
                "3. 로그인이 완료되면, 브라우저를 닫고 다시 [발행하기] 버튼을 눌러주세요."
            )
            return False

        # --- 글쓰기 ---
        st.info("로그인 상태가 확인되었습니다. 블로그 글 작성을 시작합니다...")
        wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, "mainFrame")))

        # 제목 입력
        title_input = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".se-title-input__input")))
        pyperclip.copy(title)
        title_input.click()
        title_input.clear()
        title_input.send_keys(pyperclip.paste())
        time.sleep(1)

        # 본문 입력 (pyperclip을 사용해 클립보드 기반 붙여넣기)
        content_body = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".se-main-container")))
        content_body.click()
        pyperclip.copy(content)
        content_body.send_keys(pyperclip.paste())
        time.sleep(2)

        # 태그 입력
        try:
            tag_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".btn_tag")))
            tag_button.click()
            time.sleep(1)
            tag_input = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".se-tag-input__input")))
            for tag in tags:
                tag_input.send_keys(tag.strip() + "\n")
                time.sleep(0.5)
        except Exception as tag_e:
            st.warning(f"태그 입력 중 작은 오류가 발생했으나, 발행을 계속 진행합니다: {tag_e}")

        # --- 발행 ---
        st.info("최종 발행을 진행합니다...")
        publish_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".btn_publish")))
        publish_button.click()
        time.sleep(2)

        # 최종 발행 확인 버튼 클릭
        final_publish_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".btn_confirm")))
        final_publish_button.click()
        time.sleep(5) # 발행 완료까지 대기

        return True

    except Exception as e:
        st.error(f"블로그 포스팅 중 오류가 발생했습니다: {e}")
        return False
    finally:
        if driver:
            st.info("브라우저를 종료합니다.")
            driver.quit()


# --- 3. Streamlit UI 구성 ---

st.set_page_config(page_title="AI 블로그 포스팅 자동화", layout="wide")
st.title("💼 영업사원을 위한 계약 리뷰 -> 블로그 포스팅 자동화 툴")
st.info("좌측에 계약 리뷰를 입력하고 단계별 버튼을 클릭하여 블로그 글을 완성하고 발행해보세요.")

# 세션 상태(st.session_state)를 사용하여 데이터 유지
if 'blog_title' not in st.session_state:
    st.session_state.blog_title = ""
if 'blog_content' not in st.session_state:
    st.session_state.blog_content = ""
if 'blog_tags' not in st.session_state:
    st.session_state.blog_tags = []

col1, col2 = st.columns(2)

with col1:
    st.subheader("1단계: 계약 리뷰 입력")
    sample_review = """
고객사: (주)스마트팩토리
솔루션: AI 기반 생산 공정 최적화 솔루션
핵심 문제: 잦은 설비 오류로 인한 생산 라인 중단 및 불량률 증가
기대 효과: 실시간 설비 모니터링 및 예측 정비를 통한 가동률 20% 향상 및 불량률 5% 미만 달성
    """
    contract_review = st.text_area("이곳에 계약 리뷰나 미팅 노트를 붙여넣으세요.", height=250, value=sample_review)
    
    st.subheader("2단계: 블로그 초안 생성")
    clova_studio_api_key = st.text_input("CLOVA Studio API Key", type="password", help="CLOVA Studio > 내 프로젝트 > API Key 탭에서 복사하세요.")
    
    if st.button("✨ AI로 블로그 글 생성하기", use_container_width=True):
        if not all([contract_review, clova_studio_api_key]):
            st.error("리뷰 내용과 CLOVA Studio API Key를 모두 입력해주세요.")
        else:
            with st.spinner("CLOVA AI가 블로그 초안을 작성 중입니다..."):
                title, content, tags = generate_blog_post_from_review(clova_studio_api_key, contract_review)
                st.session_state.blog_title = title
                st.session_state.blog_content = content
                st.session_state.blog_tags = tags
            
            if title != "API 호출 실패":
                st.success("블로그 초안 생성이 완료되었습니다! 우측에서 내용을 확인하고 수정하세요.")

with col2:
    st.subheader("3단계: 검토, 수정 및 발행")
    if st.session_state.blog_title:
        edited_title = st.text_input("제목", value=st.session_state.blog_title)
        edited_content = st.text_area("본문", value=st.session_state.blog_content, height=400)
        
        # 태그를 문자열로 변환하여 표시하고, 다시 리스트로 변환
        tags_str = ", ".join(st.session_state.blog_tags)
        edited_tags_str = st.text_input("태그 (쉼표로 구분)", value=tags_str)
        edited_tags = [tag.strip() for tag in edited_tags_str.split(',') if tag.strip()]

        st.markdown("---")
        st.subheader("4단계: 네이버 블로그 발행")
        
        if st.button("🚀 네이버 블로그에 발행하기", type="primary", use_container_width=True):
            with st.spinner("블로그 발행을 시작합니다... 브라우저가 실행됩니다."):
                success = post_to_naver_blog(edited_title, edited_content, edited_tags)
            if success:
                st.success("블로그 포스팅이 성공적으로 완료되었습니다!")
                st.balloons()
    else:
        st.info("좌측에서 [AI로 블로그 글 생성하기] 버튼을 누르면 여기에 초안이 표시됩니다.")
