from openai import OpenAI
import streamlit as st

def main():
    if "text" not in st.session_state:
        st.session_state.text = ""
        
    with st.sidebar:
        api_key = st.text_input("API KEY", key="_api_key")
        base_url = st.text_input("BASE URL", key="_base_url")
        model = st.text_input("Model", key="_model")
        
        clear_btn = st.button("Clear", key="_clear")
        
        max_tokens = st.slider("Max Tokens", 1, 32768, 1024, 1, key="mtokens")
        temperature = st.slider("Temperature", 0.00, 2.00, 0.60, 0.01, key="temp")
        top_p = st.slider("Top_P", 0.01, 1.00, 1.00, 0.01, key="topp")
        top_k = st.slider("Top_K", 1, 100, 50, 1, key="topk")
        f_penalty=st.slider("Frequency Penalty", -2.00, 2.00, 0.00, 0.01, key="freq")
        p_penalty = st.slider("Presence Penalty", -2.00, 2.00, 0.00, 0.01, key="pres")
        
    prompt = st.text_input("Prompt", key="_prompt")
    suffix = st.text_input("Suffix", key="_suffix")
    submit_btn = st.button("Submit", key="_submit")
    
    if api_key and base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
    
    st.markdown(st.session_state.text)
    
    if clear_btn:
        st.session_state.text = ""
        st.rerun()
        
    def gen(response):
        for chunk in response:
            if chunk.choices[0].text is not None:
                yield chunk.choices[0].text
                st.session_state.text += chunk.choices[0].text
        
    if prompt and submit_btn and suffix:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=f_penalty,
            presence_penalty=p_penalty,
            stream=True
        )
        try:
            st.write_stream(gen(response))
        except Exception as e:
            st.error(str(e))
        st.rerun()
    elif prompt and submit_btn and not suffix:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=f_penalty,
            presence_penalty=p_penalty,
            stream=True
        )
        try:
            st.write_stream(gen(response))
        except Exception as e:
            st.error(str(e))
        st.rerun()

if __name__ == "__main__":
    main()
