import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime

# Define your data and functions here, just as you have done in your code.

# Custom CSS style for the ToggleButtons
button_style = """
<style>
.widget-toggle-buttons button {
    /* Customize the background color when the button is in the "on" state */
    background-color: #2ca6a6;
    /* font color */
    color: #fff;
    /* Make the text bold */
    font-weight: bold;
    font-size: 14px;
    /* Set the width to automatically fit the text */
    width: auto;
    /* Add spacing around the buttons */
    margin: 5px;
}
</style>
"""

# Display the custom CSS style
st.markdown(button_style, unsafe_allow_html=True)

# Main function to create the dashboard
def main():
    # Add a sidebar with process stage selection
    process_stage = st.sidebar.selectbox("Select Process Stage", 
                                         ['Data Quality Evaluation', 
                                          'Performance Metrics & ML Types', 
                                          'Results Evaluation'])

    if process_stage == 'Data Quality Evaluation':
        # Add widgets for Data Quality Evaluation stage
        graph_type = st.selectbox("Select Graph Type", 
                                  ['Total Tests vs. Jobs', 'Tests Run per Build'])
        
        rawData = pd.read_csv('FATbucketPassFails.csv')
        
        if graph_type == 'Total Tests vs. Jobs':
            # Add widgets for 'Total Tests vs. Jobs'
            range_type = st.selectbox("Select Range", ['By Build', 'By Day', 'By Bucket'])
            
            if range_type == 'By Build':
                # Clear and initialize the yAxis and testFreq variables
                xAxis = []
                
                #create list of parent job names
                jobNames = rawData['PARENT_JOB_NAME'].tolist()

                #counts how often each element appears ie. total tests run for each job
                totalTests = pd.Series(jobNames).value_counts()
                totalList = totalTests.tolist()
                totalList_NDSorted= sorted([*set(totalList)]) 
                
                testFreq = []
                filteredTotalList = []

                for i, x in enumerate(totalList_NDSorted):
                    freqValue = pd.Series(totalTests).value_counts()[x]
                    if freqValue > 2:
                        testFreq.append(freqValue)
                        filteredTotalList.append(x)

                xAxis = np.arange(len(filteredTotalList))
                
                # Clear the current plot
                fig, ax = plt.subplots(1)
                ax.clear()
                plt.scatter(xAxis, testFreq, color='blue', s=10)

                # Set the titles
                ax.set_title('Total Tests vs. Number of Jobs',fontname = 'monospace', fontsize=13)
                ax.set_ylabel('Number of Jobs',fontname = 'monospace', fontsize=10)
                ax.set_xlabel('Total Tests',fontname = 'monospace', fontsize=10)

                # Set the tick positions and labels
                num_ticks = 10
                indices = np.linspace(0, len(filteredTotalList) - 1, num_ticks, dtype=int)
                ticks = [filteredTotalList[i] for i in indices]
                plt.xticks(indices, ticks)
                
                # Add a checkbox widget for the grid
                show_grid = st.checkbox("Show Grid")
                if show_grid:
                    ax.grid(True)
                else:
                    ax.grid(False)

                st.pyplot(fig)

            elif range_type == 'By Day':
                # Clear and initialize the yAxis and testFreq variables
                xAxis = []
                testFreq = []
                
                #create list of parent job names
                jobNames = rawData['PARENT_JOB_NAME'].str.slice(stop=8).tolist()

                #counts how often each element appears ie. total tests run for each job
                totalTests = pd.Series(jobNames).value_counts()
                totalList = totalTests.tolist()
                totalList_NDSorted= sorted([*set(totalList)]) 

                xAxis = np.arange(len(totalList_NDSorted))

                for x in totalList_NDSorted:
                    freqValue = pd.Series(totalTests).value_counts()[x]
                    testFreq.append(freqValue)
                    
                # Clear the current plot
                fig, ax = plt.subplots(1)
                ax.clear()
                plt.scatter(xAxis, testFreq, color='blue', s=10)

                # Set the titles
                ax.set_title('Total Tests vs. Number of Jobs',fontname = 'monospace', fontsize=13)
                ax.set_ylabel('Number of Jobs',fontname = 'monospace', fontsize=10)
                ax.set_xlabel('Total Tests',fontname = 'monospace', fontsize=10)

                # Set the tick positions and labels
                num_ticks = 10
                indices = np.linspace(0, len(totalList_NDSorted) - 1, num_ticks, dtype=int)
                ticks = [totalList_NDSorted[i] for i in indices]
                plt.xticks(indices, ticks)
                
                # Add a checkbox widget for the grid
                show_grid = st.checkbox("Show Grid")
                if show_grid:
                    ax.grid(True)
                else:
                    ax.grid(False)

                st.pyplot(fig)

            elif range_type == 'By Bucket':
                # Clear and initialize the yAxis and testFreq variables
                xAxis = []
                testFreq = []
                
                #create list of parent job names
                jobNames = rawData['BUCKET_NAME'].tolist()

                #counts how often each element appears ie. total tests run for each job
                totalTests = pd.Series(jobNames).value_counts()
                totalList = totalTests.tolist()
                totalList_NDSorted= sorted([*set(totalList)]) 

                xAxis = np.arange(len(totalList_NDSorted))

                for x in totalList_NDSorted:
                    freqValue = pd.Series(totalTests).value_counts()[x]
                    testFreq.append(freqValue)
                    
                # Clear the current plot
                fig, ax = plt.subplots(1)
                ax.clear()
                plt.scatter(xAxis, testFreq, color='blue', s=10)

                # Set the titles
                ax.set_title('Total Tests vs. Number of Jobs',fontname = 'monospace', fontsize=13)
                ax.set_ylabel('Number of Jobs',fontname = 'monospace', fontsize=10)
                ax.set_xlabel('Total Tests',fontname = 'monospace', fontsize=10)

                # Set the tick positions and labels
                num_ticks = 10
                indices = np.linspace(0, len(totalList_NDSorted) - 1, num_ticks, dtype=int)
                ticks = [totalList_NDSorted[i] for i in indices]
                plt.xticks(indices, ticks)
                
                # Add a checkbox widget for the grid
                show_grid = st.checkbox("Show Grid")
                if show_grid:
                    ax.grid(True)
                else:
                    ax.grid(False)

                st.pyplot(fig)


        elif graph_type == 'Tests Run per Build':
            # Convert 'PARENT_JOB_NAME' column to string
            rawData['PARENT_JOB_NAME'] = rawData['PARENT_JOB_NAME'].astype(str)

            # Remove rows with NaN values in 'PARENT_JOB_NAME'
            rawData = rawData.dropna(subset=['PARENT_JOB_NAME'])

            # Extract valid job names and their counts
            totalTests = rawData['PARENT_JOB_NAME'].value_counts()

            # Extract valid job names for sorting and creating date list
            valid_job_names = totalTests.index.tolist()

            # Convert date strings to datetime format, skipping 'nan' values
            date = []
            valid_totalList = []

            for d in valid_job_names:
                try:
                    date.append(datetime.strptime(d, '%Y%m%d-%H%M'))
                    valid_totalList.append(totalTests[d])
                except ValueError:
                    pass  # Skip invalid date strings

            # Create scatter plot
            fig, ax = plt.subplots()
            ax.scatter(mdates.date2num(date), valid_totalList, color='blue', s=10)

            # Display Labels
            ax.set_title('Total Tests by Build Date',fontname = 'monospace', fontsize=15)
            ax.set_xlabel('Build',fontname = 'monospace', fontsize=10)
            ax.set_ylabel('Total Tests',fontname = 'monospace', fontsize=10)


            # Set the locator and formatter for the x-axis
            locator = mdates.AutoDateLocator(minticks=8)
            formatter = mdates.DateFormatter('%Y-%m-%d %H:%M')
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            plt.xticks(rotation=45)

            plt.tight_layout()

            # Add a checkbox widget for the grid
            show_grid = st.checkbox("Show Grid")
            if show_grid:
                ax.grid(True)
            else:
                ax.grid(False)

            st.pyplot(fig)
            pass

    elif process_stage == 'Performance Metrics & ML Types':
        # Add widgets for Performance Metrics & ML Types stage
        graph_type = st.selectbox("Select Graph Type", ['Metric Evaluation', 'ML Evaluations'])

        if graph_type == 'Metric Evaluation':
            ml_type = st.selectbox("Select ML Type", ['DT', 'RF', 'GB', 'XGB', 'All'])
            
            if ml_type == 'DT':
                #read data file
                allData = pd.read_csv('classification_report_DecisionTreeClassifier.csv')

            elif ml_type == 'RF':
                #read data file
                allData = pd.read_csv('classification_report_RandomForestClassifier.csv')

            elif ml_type == 'GB':
                #read data file
                allData = pd.read_csv('classification_report_GradientBoostingClassifier.csv')

            elif ml_type == 'XGB':
                #read data file
                allData = pd.read_csv('classification_report_XGBClassifier.csv')
                
            elif ml_type == 'All':
                #DT
                allData = pd.read_csv('classification_report_DecisionTreeClassifier.csv')

                #retrieve values
                precision = allData['precision']
                recall = allData['recall']
                f1_score = allData['f1-score']

                #create rows
                data = [precision[1]*100, recall[1]*100, f1_score[1]*100]
                
                #RF
                allData1 = pd.read_csv('classification_report_RandomForestClassifier.csv')

                #retrieve values
                precision1 = allData1['precision']
                recall1 = allData1['recall']
                f1_score1 = allData1['f1-score']

                #create rows
                data1 = [precision1[1]*100, recall1[1]*100, f1_score1[1]*100]
                
                #GB
                allData2 = pd.read_csv('classification_report_GradientBoostingClassifier.csv')

                #retrieve values
                precision2 = allData2['precision']
                recall2 = allData2['recall']
                f1_score2 = allData2['f1-score']

                #create rows
                data2 = [precision2[1]*100, recall2[1]*100, f1_score2[1]*100]

                #XGB
                allData3 = pd.read_csv('classification_report_XGBClassifier.csv')

                #retrieve values
                precision3 = allData3['precision']
                recall3 = allData3['recall']
                f1_score3 = allData3['f1-score']

                #create rows
                data3 = [precision3[1]*100, recall3[1]*100, f1_score3[1]*100]
                
                # Create subplots 
                fig, axs = plt.subplots(1, 4, figsize=(15, 5))
                x_labels = ['Precision', 'Recall', 'F1 Score']
                
                bars1 = axs[0].bar(x_labels, data, width=0.5, color = 'cornflowerblue')
                axs[0].bar_label(bars1, labels=[f'{val:.2f}' for val in data], label_type='edge')
                axs[0].set_ylim(0, 100)

                bars2 = axs[1].bar(x_labels, data1, width=0.5, color = 'goldenrod')
                axs[1].bar_label(bars2, labels=[f'{val:.2f}' for val in data1], label_type='edge')
                axs[1].set_ylim(0, 100)

                bars3 = axs[2].bar(x_labels, data2, width=0.5, color = 'olivedrab')
                axs[2].bar_label(bars3, labels=[f'{val:.2f}' for val in data2], label_type='edge')
                axs[2].set_ylim(0, 100)

                bars4 = axs[3].bar(x_labels, data3, width=0.5, color = 'rebeccapurple')
                axs[3].bar_label(bars4, labels=[f'{val:.2f}' for val in data3], label_type='edge')
                axs[3].set_ylim(0, 100)
                
                axs[0].set_title("Evaluation of Failed tests using DT")
                axs[1].set_title("Evaluation of Failed tests using RF")
                axs[2].set_title("Evaluation of Failed tests using GB")
                axs[3].set_title("Evaluation of Failed tests using XGB")
                
                plt.tight_layout()
                st.pyplot(fig)


            if ml_type != 'All':

                #retrieve values
                precision = allData['precision']
                recall = allData['recall']
                f1_score = allData['f1-score']

                #create rows
                data = [precision[1]*100, recall[1]*100, f1_score[1]*100]
                x_labels = ['Precision', 'Recall', 'F1 Score']

                # Create a figure and axis
                fig, ax = plt.subplots()

                # Plot the bar chart
                bars = ax.bar(x_labels, data, width=0.4, color='cornflowerblue')

                #   Add value labels to each bar
                ax.bar_label(bars, labels=[f'{val:.2f}' for val in data], label_type='edge')
                ax.set_ylim(0, 100)

                # Set title
                ax.set_title('Evaluation of Failed tests using ' + ml_type)

                # Show the plot
                st.pyplot(fig)

        elif graph_type == 'ML Evaluations':
            evaluation_type = st.selectbox("Select Evaluation Type", ['Recall', 'Precision', 'F1 Score', 'All'])
            
            dtData = pd.read_csv('classification_report_DecisionTreeClassifier.csv')
            rfData = pd.read_csv('classification_report_RandomForestClassifier.csv')
            gbData = pd.read_csv('classification_report_GradientBoostingClassifier.csv')
            xgbData = pd.read_csv('classification_report_XGBClassifier.csv')

            if evaluation_type == 'Recall':
                dt = dtData['recall']
                rf = rfData['recall']
                gb = gbData['recall']
                xgb = xgbData['recall']

                data = [dt[3]*100, rf[3]*100, gb[3]*100, xgb[3]*100]

            elif evaluation_type == 'Precision':
                dt = dtData['precision']
                rf = rfData['precision']
                gb = gbData['precision']
                xgb = xgbData['precision']

                data = [dt[3]*100, rf[3]*100, gb[3]*100, xgb[3]*100]
                
                
                
            elif evaluation_type == 'F1 Score':
                dt = dtData['f1-score']
                rf = rfData['f1-score']
                gb = gbData['f1-score']
                xgb = xgbData['f1-score']

                data = [dt[1]*100, rf[1]*100, gb[1]*100, xgb[1]*100]
                

            elif evaluation_type == 'All':
                #recall
                dt1 = dtData['recall']
                rf1 = rfData['recall']
                gb1 = gbData['recall']
                xgb1 = xgbData['recall']

                data1 = [dt1[1]*100, rf1[1]*100, gb1[1]*100, xgb1[1]*100]

                #precision
                dt2 = dtData['precision']
                rf2 = rfData['precision']
                gb2 = gbData['precision']
                xgb2 = xgbData['precision']

                data2 = [dt2[1]*100, rf2[1]*100, gb2[1]*100, xgb2[1]*100]

                #f1_score
                dt3 = dtData['f1-score']
                rf3 = rfData['f1-score']
                gb3 = gbData['f1-score']
                xgb3 = xgbData['f1-score']

                data3 = [dt3[1]*100, rf3[1]*100, gb3[1]*100, xgb3[1]*100]
                
                # Create subplots 
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                x_labels = ['DT', 'RF', 'GB', 'XGB']

                bars1 = axs[0].bar(x_labels, data1, width=0.5, color = 'cornflowerblue')
                axs[0].bar_label(bars1, labels=[f'{val:.2f}' for val in data1], label_type='edge')
                axs[0].set_ylim(0, 100)

                bars2 = axs[1].bar(x_labels, data2, width=0.5, color = 'olivedrab')
                axs[1].bar_label(bars2, labels=[f'{val:.2f}' for val in data2], label_type='edge')
                axs[1].set_ylim(0, 100)

                bars3 = axs[2].bar(x_labels, data3, width=0.5, color = 'rebeccapurple')
                axs[2].bar_label(bars3, labels=[f'{val:.2f}' for val in data3], label_type='edge')
                axs[2].set_ylim(0, 100)
                
                axs[0].set_title("Recall for Failed Test Suites")
                axs[1].set_title("Precision for Failed Test Suites")
                axs[2].set_title("F1 Score for Failed Test Suites")
                
                plt.tight_layout()
                st.pyplot(fig)


            if evaluation_type != 'All':

                x_labels = ['DT', 'RF', 'GB', 'XGB']

                # Create a figure and axis
                fig, ax = plt.subplots()

                # Plot the bar chart
                bars = ax.bar(x_labels, data, width=0.5, color='cornflowerblue')

                # Add value labels to each bar
                ax.bar_label(bars, labels=[f'{val:.2f}' for val in data], label_type='edge')
                ax.set_ylim(0, 100)

                # Set title
                ax.set_title(evaluation_type + ' for Failed Test Suites')

                # Show the plot
                st.pyplot(fig)

    elif process_stage == 'Results Evaluation':
        # Add widgets for Results Evaluation stage
        results_eval = st.selectbox("Select Results Evaluation", ['AFP', 'APFD'])
        
        if results_eval == 'AFP':
            ml_choice = st.selectbox("Select ML Type", ['DT', 'RF', 'GB', 'XGB'])
            fp_type = st.selectbox("Select Failure Position Type", ['FFP', 'AFP', 'LFP'])
            # Add your plot code here using matplotlib or other plotting libraries.

            # read each ml data file
            dtData = pd.read_csv('DecisionTreeClassifier_afpcleaned.csv')

            gbData = pd.read_csv('GradientBoostingClassifier_afpcleaned.csv')

            rfData = pd.read_csv('RandomForestClassifier_afpcleaned.csv')

            xgbData = pd.read_csv('XGBClassifier_afpcleaned.csv')

            #read base data file
            actualData = pd.read_csv('base_2023_afp.csv')

            afpActual = actualData.averag_failure_position

            # Get the 'parent_job_name' column as date strings
            date_strings = actualData['parent_job_name']

            # Convert date strings to datetime format
            date = [datetime.strptime(d, '%Y%m%d-%H%M') for d in date_strings]

            if ml_choice == 'DT':
                selected_data = dtData
            elif ml_choice == 'RF':
                selected_data = rfData
            elif ml_choice == 'GB':
                selected_data = gbData
            elif ml_choice == 'XGB':
                selected_data = xgbData

            # Add a slider widget to control the date range
            min_date = min(date)
            max_date = max(date)
            selected_range = st.slider("Select Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date))
            
            # Create a new filtered_date based on the selected date range
            filtered_date = [d for d in date if selected_range[0] <= d <= selected_range[1]]

            # Create a new filtered_afpActual based on the selected date range
            filtered_afpActual = [afp for d, afp in zip(date, afpActual) if selected_range[0] <= d <= selected_range[1]]
            
            # Convert the selected_range tuple to a list of dates
            selected_range_list = list(selected_range)

            fig, ax = plt.subplots(1, figsize=(10, 6))
            ax.plot(mdates.date2num(filtered_date), filtered_afpActual, label='Without ML')
            ax.fill_between(mdates.date2num(filtered_date), filtered_afpActual, 0, facecolor='C9', alpha=0.4)
            ax.legend(loc='upper left')

            # Display Labels
            ax.set_title(f'{fp_type} Graph', fontname='monospace', fontsize=15)
            ax.set_xlabel('Date', fontname='monospace', fontsize=10)
            ax.set_ylabel(f'{fp_type} Values', fontname='monospace', fontsize=10)

            # Set the locator and formatter for the x-axis
            locator = mdates.AutoDateLocator(minticks=8)
            formatter = mdates.DateFormatter('%Y-%m-%d')
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            plt.xticks(rotation=45)

            # Convert date list to numpy array for comparison
            date_array = np.array(date)
            
            # Apply the mask based on the selected date range
            mask = (date_array >= selected_range_list[0]) & (date_array <= selected_range_list[1])

            # Add a checkbox widget for the grid for this specific subplot
            show_grid = st.checkbox("Show Grid", key=f"{results_eval}_{ml_choice}_{fp_type}")
            if show_grid:
                ax.grid(True)
            else:
                ax.grid(False)

            if fp_type == 'FFP':
                ax.plot(mdates.date2num(filtered_date), selected_data.first_failure[mask], label='FFP', color='C2')
                ax.fill_between(mdates.date2num(filtered_date), selected_data.first_failure[mask], 0, facecolor='C2', alpha=0.4)

                # Checkbox to fill between FFP and AFP actual where FFP is higher
                mask_afp_higher = np.array(filtered_afpActual) < np.array(selected_data.first_failure[mask])

                show_pink_fill = st.checkbox("Show Performance Difference", key=f"{results_eval}_{ml_choice}_{fp_type}_FFP")
                if show_pink_fill:
                    ax.fill_between(mdates.date2num(filtered_date), selected_data.first_failure[mask], filtered_afpActual,
                                    where=mask_afp_higher, facecolor='magenta', alpha=0.4)

            elif fp_type == 'LFP':
                ax.plot(mdates.date2num(filtered_date), selected_data.last_failure[mask], label='LFP', color='C3')
                ax.fill_between(mdates.date2num(filtered_date), selected_data.last_failure[mask], 0, facecolor='C3', alpha=0.4)

                # Checkbox to fill between LFP and AFP actual where LFP is higher
                mask_afp_higher = np.array(filtered_afpActual) < np.array(selected_data.last_failure[mask])

                show_pink_fill = st.checkbox("Show Performance Difference", key=f"{results_eval}_{ml_choice}_{fp_type}_LFP")
                if show_pink_fill:
                    ax.fill_between(mdates.date2num(filtered_date), selected_data.last_failure[mask], filtered_afpActual,
                                    where=mask_afp_higher, facecolor='magenta', alpha=0.4)

            elif fp_type == 'AFP':
                ax.plot(mdates.date2num(filtered_date), selected_data.averag_failure_position[mask], label='AFP', color='C4')
                ax.fill_between(mdates.date2num(filtered_date), selected_data.averag_failure_position[mask], 0, facecolor='C4', alpha=0.4)

                # Checkbox to fill between AFP and AFP actual where AFP is higher
                mask_afp_higher = np.array(filtered_afpActual) < np.array(selected_data.averag_failure_position[mask])

                show_pink_fill = st.checkbox("Show Performance Difference", key=f"{results_eval}_{ml_choice}_{fp_type}_AFP")
                if show_pink_fill:
                    ax.fill_between(mdates.date2num(filtered_date), selected_data.averag_failure_position[mask], filtered_afpActual,
                                    where=mask_afp_higher, facecolor='magenta', alpha=0.4)

            st.pyplot(fig)
            
            # Clear the figure and axis
            plt.close(fig)

        elif results_eval == 'APFD':
            #Read Data File
            apfdData = pd.read_csv('apfdGraphFile.csv')

            #Extract specific Data            
            dtValue = apfdData['DecisionTreeClassifier']
            rfValue = apfdData['RandomForestClassifier']
            gbValue = apfdData['GradientBoostingClassifier']
            xgbValue = apfdData['XGBClassifier']
            
            #create rows
            data = [dtValue[0]*100, rfValue[0]*100, gbValue[0]*100, xgbValue[0]*100]
            
            # Create subplots 
            fig, axs = plt.subplots(1)
            x_labels = ['DT', 'RF', 'GB', 'XGB']
            
            #Plot Graph
            bars = axs.bar(x_labels, data, width=0.5, color = 'cornflowerblue')
            axs.bar_label(bars, labels=[f'{val:.2f}' for val in data], label_type='edge')
            axs.set_ylim(0, 100)
            axs.set_title("Evaluation of APFD Value for Different ML Models")
            
            plt.tight_layout()
            st.pyplot(fig)
            
            pass

# Run the main function to start the streamlit app
if __name__ == "__main__":
    main()
