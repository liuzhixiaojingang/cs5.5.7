import streamlit as st
import joblib
import numpy as np

# 加载模型
@st.cache_resource
def load_model():
    model_path = "best_mlp_model.pkl"
    return joblib.load(model_path)

# 主函数
def main():
    st.title("烧伤程度识别系统")
    st.write("请输入20个DL特征值进行烧伤程度预测 (0-5)")

    # 创建输入表单
    with st.form("input_form"):
        inputs = []
        cols = st.columns(4)  # 创建4列布局
        
        for i in range(20):
            with cols[i % 4]:  # 均匀分配到各列
                inputs.append(st.number_input(f"DL{i+1}", key=f"dl{i}"))
        
        submitted = st.form_submit_button("预测")
    
    # 当用户提交表单时进行预测
    if submitted:
        model = load_model()
        input_array = np.array(inputs).reshape(1, -1)
        prediction = model.predict(input_array)
        
        st.success(f"预测结果: 烧伤程度 {prediction[0]}")

if __name__ == "__main__":
    main()