import psutil
import streamlit as st


def render():
    st.header("System Monitor")
    st.caption("Real-time server and resource metrics")

    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    net = psutil.net_io_counters()

    col1, col2 = st.columns(2)
    col1.metric("CPU %", f"{cpu}")
    col1.metric("RAM %", f"{mem}")
    col2.metric("Bytes Sent", f"{net.bytes_sent}")
    col2.metric("Bytes Recv", f"{net.bytes_recv}")

    if cpu > 80 or mem > 80:
        st.error("Resource usage high!")
    else:
        st.success("Server health OK")


