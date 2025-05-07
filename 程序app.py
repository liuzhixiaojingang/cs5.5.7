import streamlit as st
import pickle
import pandas as pd

# 模型加载函数
@st.cache_resource
def load_model():
    with open('best_mlp_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def main():
    # 加载训练好的模型
    model = load_model()
    
    # 设置页面标题
    st.title('🩺 烧伤程度智能识别系统')
    st.markdown("---")
    
    # 创建参数输入界面
    st.header('参数输入区')
    cols = st.columns(4)  # 创建4列布局
    
    input_data = {}
    for i in range(20):
        with cols[i//5]:  # 将参数分配到4列显示
            input_data[f'DL{i+1}'] = st.number_input(
                f'DL{i+1}', 
                value=0.0,
                format="%.4f",
                key=f'dl_{i}'
            )
    
    # 转换为DataFrame格式
    input_df = pd.DataFrame([input_data])
    
    # 预测按钮
    st.markdown("---")
    if st.button('🚀 开始预测'):
        try:
            prediction = model.predict(input_df)
            st.success(f'### 预测结果：{prediction[0]}')
        except Exception as e:
            st.error(f'预测错误：{str(e)}')

if __name__ == '__main__':
    main()