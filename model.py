import numpy as np
import tensorflow as  tf
import pandas as pd
import streamlit as st

#st.set_option('depreciation.showfileUploaderEncoading', False)
@st.cache(allow_output_mutation=True)

def load_model():
    model = tf.keras.models.load_model("SOC_dynamic_dropout_v21.h5")

    return model
model= load_model()

def load_model2():
    model2= tf.keras.models.load_model("Nissan Leaf-1.h5")
    return model2
model2= load_model2()


def predict_soc(hour, minute, speed, voltage, current, temp):
    prediction= model.predict([[hour, minute, speed, voltage, current, temp]])
    print(prediction)
    return prediction

def energy_con(speed, accel):
    prediction= model2.predict([[speed, accel]])
    print(prediction)
    return prediction

# Python code to get the Cumulative sum of a list
def Cumulative(lists):
    cu_list = []
    length = len(lists)
    cu_list = [sum(lists[0:x:1]) for x in range(0, length+1)]
    return cu_list[1:]



def main():
    st.title("Nissan Leaf ")
    html_temp="""
    <div style="bacground-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit SOC Predictor ML App</h2>
    </div>
    """

    import_data= st.file_uploader("Upload driving Cycle data", type='csv')

    df= pd.read_csv(import_data)
    st.write(df)

    st.markdown(html_temp,unsafe_allow_html=True)
    # day = st.text_input("Day")

    ini_soc= st.number_input("Initial SoC (decimals)")
    bat_cap = st.number_input("Vehicle Battery Capacity (kWh)")

    srt_energy = ini_soc * bat_cap

    output= []
    cum_energy= []





    result=""
    if st.button("Submit"):

        # result = str(result)[2:-2]


        # result2 = str(result2)[2:-2]

        # st.success("Current SoC is {0:.5s} %".format(result))
        # st.success("Instantaneous Energy Consumption is {0:.4s} kWh".format(result2))

        for i in range(len(df)):
            # hour = df.hour.iloc[i].astype('float')
            # hour= df.hour.astype('float')
            # (st.slider("Hour", 8, 19, step=1))
            # hour = pd.DataFrame(hour).to_numpy()
            # hour = hour.reshape(hour.shape[0], 1, hour.shape[1])

            # minute = df.minute.iloc[i].astype('float')
            # minute = df.minute.astype('float')
            # (st.slider("Minute", 0, 60, step=1))

            speed = df.speed_kmph.iloc[i].astype('float')
            # speed = df.speed_Kmph.astype('float')
            # (st.number_input("Speed (Km/hr)"))
            # speed = pd.DataFrame(speed).to_numpy()
            # speed = speed.reshape(speed.shape[0], 1, speed.shape[1])

            accel = df.accel_meters_ps.iloc[i].astype('float')
            # accel = df.accel_meters_ps.astype('float')
            # (st.number_input("Acceleration (m/S^2)"))

            # voltage = df.voltage.iloc[i].astype('float')
            # voltage = df.voltage.astype('float')
            # st.slider("Voltage (V)",50.5,56.0)
            # voltage = pd.DataFrame(voltage).to_numpy()
            # voltage = voltage.reshape(voltage.shape[0], 1, voltage.shape[1])

            # current = df.current.iloc[i].astype('float')
            # current = df.current.astype('float')
            # st.number_input("Current (A)")
            # current = pd.DataFrame(current).to_numpy()
            # current = current.reshape(current.shape[0], 1, current.shape[1])

            # temp = df.temperature.iloc[i].astype('float')
            # temp = df.temperature.astype('float')

            # st.number_input("Temp (°C)")
            # temp = pd.DataFrame(temp).to_numpy()
            # temp = temp.reshape(temp.shape[0], 1, temp.shape[1])
            # df= np.array([hour, minute, speed, voltage, current, temp])
            # df = df.reshape(1, 1, df.shape[0])
            # st.write(df.shape)

            # result = predict_soc(hour, minute, speed, voltage, current, temp)
            result2 = energy_con(speed, accel)
            # cum_energy= Cumulative(result2)
            bat_energy= srt_energy - (result2/3600)
            soc = ((bat_energy) / bat_cap) * 100
            srt_energy = bat_energy

            output.append((result2, bat_energy, soc))
        df1 = pd.DataFrame(output, columns=['Energy Consumption (kWh)','Battery Energy (kWh)', 'SoC (%)'])
        df1 = df1.astype('float')
        # st.write(df1.dtypes)
        df1.to_excel(r'/Users/ameya/Desktop/Fastsim ML/outpt.xlsx', index=False)

        st.write(df1.style.format({ 'Energy Consumption (kWh)': '{:.2f}', 'Battery Energy (kWh)': '{:.2f}', 'SoC (%)': "{:.2f}%",}))

if __name__ == '__main__':
    main()