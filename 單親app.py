import streamlit as st
import time
import random
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import base64
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="單親互助平台", layout="wide", initial_sidebar_state="collapsed")

# 多語言字典
TEXTS = {
    "zh": {
        "title": "單親互助平台",
        "slogan": "一個人顧囝仔，毋免驚孤單！",
        "welcome_home": "歡迎回家",
        "connect_love": "串聯力量，延續愛",
        "google_login": "使用 Google 登入",
        "fb_login": "使用 Facebook 登入",
        "apple_login": "使用 Apple 登入",
        "or": "或",
        "email": "電子郵件",
        "start_journey": "開啟溫暖之旅",
        "welcome": "歡迎回家！你的積分：",
        "points": "積分：{} 點",
        "hi": "嗨，{}",
        "logout": "登出",
        "language": "語言",
        "menu_home": "首頁",
        "menu_circle": "生活圈",
        "menu_match": "匹配",
        "menu_resources": "資源中心",
        "menu_tips": "育兒小教室",
        "menu_profile": "個人設定",
        "verify_prompt": "建議到「個人設定」完成驗證，解鎖完整功能～",
        "upgrade": "升級會員",
        "redeem": "積分兌換",
        "nearby": "附近親子活動",
        "post_content": "想分享什麼？",
        "upload_image": "上傳圖片（選填）",
        "submit_post": "發布",
        "like": "讚",
        "reply": "回覆",
        "edit_profile": "編輯個人資料",
        "nickname": "暱稱",
        "bio": "自我介紹",
        "children": "孩子年齡（用逗號分隔）",
        "save": "儲存",
    },
    "en": {
        "title": "Single Parent Support Platform",
        "slogan": "You're not alone in raising your kids!",
        "welcome_home": "Welcome Home",
        "connect_love": "Connect Strength, Continue Love",
        "google_login": "Sign in with Google",
        "fb_login": "Sign in with Facebook",
        "apple_login": "Sign in with Apple",
        "or": "or",
        "email": "Email",
        "start_journey": "Start Your Warm Journey",
        "welcome": "Welcome home! Your points: ",
        "points": "Points: {} pts",
        "hi": "Hi, {}",
        "logout": "Logout",
        "language": "Language",
        "menu_home": "Home",
        "menu_circle": "Community",
        "menu_match": "Match",
        "menu_resources": "Resources",
        "menu_tips": "Parenting Tips",
        "menu_profile": "My Settings",
        "verify_prompt": "Suggest completing verification in 'My Settings' to unlock full features!",
        "upgrade": "Upgrade Membership",
        "redeem": "Redeem Points",
        "nearby": "Nearby Parent-Child Events",
        "post_content": "What do you want to share?",
        "upload_image": "Upload Image (optional)",
        "submit_post": "Post",
        "like": "Like",
        "reply": "Reply",
        "edit_profile": "Edit Profile",
        "nickname": "Nickname",
        "bio": "Bio",
        "children": "Children's Ages (comma separated)",
        "save": "Save",
    }
}

# CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&display=swap" rel="stylesheet">
<style>
    body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #FFF5F8, #FFE8F0, #FFF1F8) !important;
        font-family: 'Noto Sans TC', sans-serif !important;
        color: #1F2937 !important;
    }
    .login-container {
        max-width: 420px;
        margin: 80px auto;
        padding: 48px 32px;
        background: white;
        border-radius: 32px;
        box-shadow: 0 20px 60px rgba(236,72,153,0.2);
        text-align: center;
    }
    .login-title { font-size: 3rem; font-weight: bold; color: #BE185D; margin-bottom: 8px; }
    .login-subtitle { font-size: 1.2rem; color: #831843; margin-bottom: 48px; }
    .btn-login {
        width: 100%;
        padding: 16px;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 9999px !important;
        margin: 12px 0 !important;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        border: none !important;
    }
    .btn-google { background: white !important; color: #1F2937 !important; border: 1px solid #D1D5DB !important; }
    .btn-fb { background: #1877F2 !important; color: white !important; }
    .btn-apple { background: black !important; color: white !important; }
    .btn-start { background: linear-gradient(135deg, #EC4899, #DB2777) !important; color: white !important; margin-top: 32px !important; }
    .or-divider { display: flex; align-items: center; margin: 32px 0; color: #9CA3AF; }
    .or-divider::before, .or-divider::after { content: ''; flex: 1; height: 1px; background: #E5E7EB; }
    .or-divider span { padding: 0 24px; }
    .card { background: white; border-radius: 24px; padding: 24px; margin: 20px 0; box-shadow: 0 8px 32px rgba(236,72,153,0.18); border: 1px solid #FFE4EC; }
    button { background: linear-gradient(135deg, #EC4899, #DB2777) !important; color: white !important; border-radius: 20px !important; padding: 16px !important; font-size: 1.2rem !important; }
</style>
""", unsafe_allow_html=True)

# 初始化 session_state
for k, v in {
    "authenticated": False,
    "splash_shown": False,
    "lang": "zh",
    "posts": [],
    "points": 50,
    "username": "",
    "certified": False,
    "user_profile": {"nickname": "", "bio": "", "children": [], "avatar": None},
    "current_board": None,
    "chat_history": {},
    "current_chat_partner": None,
    "matches": [],
    "model": None,
    "last_post_count": 0,
    "activities": [],  # 親子活動列表
    "exchanges": []    # 積分兌換紀錄
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Splash 畫面
if not st.session_state.splash_shown:
    st.markdown(f"""
    <div style="position:fixed; inset:0; background: linear-gradient(rgba(255,245,248,0.9), rgba(255,232,240,0.9)), #FFF5F8; display:flex; align-items:center; justify-content:center; z-index:9999;">
        <div style="background:white; padding:3rem; border-radius:30px; text-align:center; box-shadow:0 15px 40px rgba(153,27,74,0.2);">
            <h1 style="font-size:4rem; color:#BE185D; margin:0;">{TEXTS[st.session_state.lang]['title']}</h1>
            <p style="font-size:2rem; color:#831843; margin-top:1rem;">{TEXTS[st.session_state.lang]['slogan']}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(1.5)
    st.session_state.splash_shown = True
    st.rerun()

# 登入頁面
if not st.session_state.authenticated:
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    
    st.markdown(f"<h1 class='login-title'>{TEXTS[st.session_state.lang]['welcome_home']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='login-subtitle'>{TEXTS[st.session_state.lang]['connect_love']}</p>", unsafe_allow_html=True)

    if st.button(f"G {TEXTS[st.session_state.lang]['google_login']}", key="google_login", use_container_width=True):
        st.session_state.authenticated = True
        st.session_state.username = "Google用戶"
        st.session_state.points = 50
        st.rerun()

    if st.button(f"f {TEXTS[st.session_state.lang]['fb_login']}", key="fb_login", use_container_width=True):
        st.session_state.authenticated = True
        st.session_state.username = "Facebook用戶"
        st.session_state.points = 50
        st.rerun()

    if st.button(f" {TEXTS[st.session_state.lang]['apple_login']}", key="apple_login", use_container_width=True):
        st.session_state.authenticated = True
        st.session_state.username = "Apple用戶"
        st.session_state.points = 50
        st.rerun()

    st.markdown("<div class='or-divider'><span>或</span></div>", unsafe_allow_html=True)

    email = st.text_input(TEXTS[st.session_state.lang]["email"], placeholder="輸入電子郵件")
    
    if st.button(TEXTS[st.session_state.lang]["start_journey"], key="start_journey", use_container_width=True):
        if email:
            st.session_state.authenticated = True
            st.session_state.username = email.split('@')[0] or "溫暖用戶"
            st.session_state.points = 50
            st.success("歡迎加入！開啟溫暖之旅～")
            st.rerun()
        else:
            st.warning("請輸入電子郵件")

    st.markdown("</div>", unsafe_allow_html=True)

else:
    # 側邊欄導航
    st.sidebar.title(TEXTS[st.session_state.lang]["title"])
    st.sidebar.markdown(f"<p style='font-size:1.2rem;'>{TEXTS[st.session_state.lang]['hi'].format(st.session_state.username)}</p>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<p style='background:#FFF1F8; padding:12px; border-radius:12px; text-align:center;'>{TEXTS[st.session_state.lang]['points'].format(st.session_state.points)}</p>", unsafe_allow_html=True)

    lang = st.sidebar.radio(TEXTS[st.session_state.lang]["language"], ["繁體中文", "English"])
    st.session_state.lang = "zh" if lang == "繁體中文" else "en"

    page = st.sidebar.radio("導航", [
        TEXTS[st.session_state.lang]["menu_home"],
        TEXTS[st.session_state.lang]["menu_circle"],
        TEXTS[st.session_state.lang]["menu_match"],
        TEXTS[st.session_state.lang]["menu_resources"],
        TEXTS[st.session_state.lang]["menu_tips"],
        TEXTS[st.session_state.lang]["menu_profile"]
    ])

    if st.sidebar.button(TEXTS[st.session_state.lang]["logout"]):
        st.session_state.authenticated = False
        st.session_state.splash_shown = False
        st.rerun()

    st.markdown(f"<h1 style='margin-bottom:1.5rem; color:#BE185D;'>{page}</h1>", unsafe_allow_html=True)

    if not st.session_state.certified and page != TEXTS[st.session_state.lang]["menu_profile"]:
        st.warning(TEXTS[st.session_state.lang]["verify_prompt"])

    # 首頁
    if page == TEXTS[st.session_state.lang]["menu_home"]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write(f"{TEXTS[st.session_state.lang]['welcome']}{st.session_state.points} 點")
        cols = st.columns(2)
        with cols[0]:
            if st.button(TEXTS[st.session_state.lang]["redeem"], use_container_width=True):
                st.info("積分兌換功能（可擴充禮品列表）")
        with cols[1]:
            if st.button(TEXTS[st.session_state.lang]["nearby"], use_container_width=True):
                st.info("附近親子活動（可擴充地圖或列表）")
        st.markdown("</div>", unsafe_allow_html=True)

    # 生活圈
    elif page == TEXTS[st.session_state.lang]["menu_circle"]:
        boards = [
            ("暖心餐桌", "🍲 輪流煮飯・團購", "warm_table"),
            ("愛心流轉", "👕 二手童裝交換", "love_flow"),
            ("共居計畫", "🏠 合租室友", "co_live"),
            ("技能交換", "📚 家教互補・補助查詢", "skill_swap"),
            ("假期不孤單", "🎈 親子活動", "holiday")
        ]

        if st.session_state.current_board is None:
            cols = st.columns(2)
            for i, (title, desc, key) in enumerate(boards):
                with cols[i % 2]:
                    st.markdown(f"<div class='card'><h3>{title}</h3><p>{desc}</p>", unsafe_allow_html=True)
                    if st.button("進入", key=f"enter_{key}"):
                        st.session_state.current_board = key
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            # 正確解包
            selected = next((title, desc, key) for title, desc, key in boards if key == st.session_state.current_board)
            title, desc, board_key = selected
            emoji = desc.split(" ")[0]

            st.subheader(f"{emoji} {title} 討論區")
            if st.button("← 返回"):
                st.session_state.current_board = None
                st.rerun()

            # 貼文列表
            board_posts = [p for p in st.session_state.posts if p.get("board") == board_key]
            for idx, post in enumerate(board_posts):
                with st.expander(f"{post['username']} • {post.get('time', '剛剛')}"):
                    if post.get("image"):
                        st.image(post["image"], use_column_width=True)
                    st.write(post["content"])
                    cols = st.columns(2)
                    with cols[0]:
                        if st.button(f"{TEXTS[st.session_state.lang]['like']} ({post.get('likes', 0)})", key=f"like_{idx}"):
                            post["likes"] = post.get("likes", 0) + 1
                            st.rerun()
                    with cols[1]:
                        reply = st.text_input("回覆...", key=f"reply_input_{idx}")
                        if st.button(TEXTS[st.session_state.lang]["reply"], key=f"reply_btn_{idx}"):
                            if reply:
                                post.setdefault("replies", []).append({"user": st.session_state.username, "text": reply})
                                st.success("已回覆！")
                                st.rerun()

            # 發新貼文 + 上傳圖片
            with st.form(key=f"post_form_{board_key}"):
                content = st.text_area(TEXTS[st.session_state.lang]["post_content"], height=120)
                uploaded_file = st.file_uploader(TEXTS[st.session_state.lang]["upload_image"], type=["jpg", "png", "jpeg"])
                submitted = st.form_submit_button(TEXTS[st.session_state.lang]["submit_post"])
                if submitted and content:
                    new_post = {
                        "board": board_key,
                        "username": st.session_state.username,
                        "content": content,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "likes": 0,
                        "replies": []
                    }
                    if uploaded_file:
                        bytes_data = uploaded_file.getvalue()
                        new_post["image"] = bytes_data
                    st.session_state.posts.append(new_post)
                    st.success("已發布！")
                    st.rerun()

    # 匹配 - 修正地圖錯誤
    elif page == TEXTS[st.session_state.lang]["menu_match"]:
        st.subheader("AI 精準匹配 + 距離雷達")
        if not st.session_state.user_profile:
            st.session_state.user_profile = {
                "children_ages": st.multiselect("小孩年齡", [1,2,3,4,5,6,7,8,9,10,11,12]),
                "preferences": {
                    "pet_friendly": st.checkbox("接受寵物"),
                    "night_shift_ok": st.checkbox("接受大夜班"),
                    "cleanliness": st.selectbox("整潔程度", ["高", "中", "低"]),
                    "parenting_style": st.selectbox("育兒風格", ["嚴格", "放鬆", "平衡"]),
                    "has_car": st.checkbox("有車")
                }
            }
        if st.button("開始匹配"):
            matches = [
                {"name": "小美媽媽", "score": 88, "desc": "高雄三民區，孩子6歲，距離 2.5km"},
                {"name": "阿強爸", "score": 75, "desc": "高雄左營區，孩子5歲，距離 4.8km"}
            ]
            for m in matches:
                st.markdown(f"<div class='card'>匹配度 {m['score']}% - {m['name']}<br>{m['desc']}</div>", unsafe_allow_html=True)

            # 修正地圖：使用 pd.DataFrame + 正確欄位 'lat' / 'lon'
            map_data = pd.DataFrame({
                'lat': [22.6273, 22.6651],
                'lon': [120.3014, 120.3051]
            })
            st.map(map_data)

    # 資源中心
    elif page == TEXTS[st.session_state.lang]["menu_resources"]:
        st.subheader("資源中心")
        st.markdown("- [單親培力計劃](https://www.sfaa.gov.tw/SFAA/Pages/List.aspx?nodeid=768)")
        st.markdown("- [特殊境遇家庭扶助](https://www.gov.tw/News_Content_26_694361)")
        st.markdown("- [單親補助指南](https://premium.parenting.com.tw/article/5093204)")
        st.markdown("- 福利諮詢專線：1957")
        st.markdown("- [法律輔助](https://www.law.org.tw/)")

    # 育兒小教室
    elif page == TEXTS[st.session_state.lang]["menu_tips"]:
        st.subheader("育兒小教室")
        tips = [
            "每天10分鐘親子遊戲，增進情感連結",
            "多蔬果、少加工食品，幫助孩子健康成長",
            "壓力大時深呼吸，也可尋求支持",
            "每天固定閱讀時間，培養學習興趣",
            "教孩子辨識陌生人，安全第一"
        ]
        for tip in tips:
            st.markdown(f"- {tip}")

    # 個人設定
    elif page == TEXTS[st.session_state.lang]["menu_profile"]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        # 大頭貼上傳
        st.subheader("大頭貼")
        uploaded_avatar = st.file_uploader("上傳大頭貼", type=["jpg", "png", "jpeg"])
        if uploaded_avatar:
            img = Image.open(uploaded_avatar)
            st.image(img, width=150, caption="已上傳大頭貼")
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            st.session_state.user_profile["avatar"] = base64.b64encode(buffered.getvalue()).decode()

        # 編輯資料
        with st.expander("編輯個人資料"):
            nickname = st.text_input("暱稱", value=st.session_state.user_profile.get("nickname", st.session_state.username))
            bio = st.text_area("自我介紹", value=st.session_state.user_profile.get("bio", ""))
            children = st.text_input("孩子年齡（用逗號分隔）", value=", ".join(map(str, st.session_state.user_profile.get("children", []))))
            privacy = st.selectbox("資料公開程度", ["完全公開", "僅匹配對象", "僅好友", "私人"])
            if st.button("儲存"):
                st.session_state.user_profile["nickname"] = nickname
                st.session_state.user_profile["bio"] = bio
                try:
                    st.session_state.user_profile["children"] = [int(x.strip()) for x in children.split(",") if x.strip()]
                except:
                    st.error("孩子年齡請輸入數字")
                st.session_state.user_profile["privacy"] = privacy
                st.success("已儲存！")

        st.write(f"名稱：{st.session_state.user_profile.get('nickname', st.session_state.username)}")
        st.write(f"積分：{st.session_state.points} 點")
        if st.session_state.certified:
            st.success("已驗證")
        else:
            if st.button("驗證身分"):
                st.session_state.certified = True
                st.success("驗證成功！")

        st.markdown("</div>", unsafe_allow_html=True)