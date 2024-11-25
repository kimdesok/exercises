import streamlit as st

import subprocess

def run_rebel_script():
    result = subprocess.run(["python3", "/home/work/exercises/streamlit-test.py"], capture_output=True, text=True)
    return result.stdout, result.stderr

output, error = run_rebel_script()
st.write("Rebel output:", output)
if error:
    st.write("Rebel error:", error)

df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df

st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))