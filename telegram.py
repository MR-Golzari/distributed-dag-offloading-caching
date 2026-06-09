import telepot
import numpy as np
import shutil
import os
from telepot.loop import MessageLoop



telegramTOKEN = '5695741795:AAEhW_fX7sVAGolFamGgCZbi8MPii_l0MQ4'
#telegramTOKEN = "6138472212:AAGZg2VbeAiI5jRC_62yOxB3MgPnlL8uvas"
#telegramTOKEN = '6138472212:AAGZg2VbeAiI5jRC_62yOxB3MgPnlL8uvas'
chat_id = 63726018
sendtotelegram = False
class Telegram:
    def __init__(self) -> None:
        self.telegrambot = telepot.Bot(telegramTOKEN)
        
        # Start the bot and listen for incoming messages
        if sendtotelegram:
            MessageLoop(self.telegrambot, {'chat': self.on_message}).run_as_thread()
    def send_running(self):
        if(sendtotelegram):
            try:
                self.running_message = self.telegrambot.sendMessage(chat_id, '~~~~~~~~Running~~~~~~')
            except Exception as e:
                exception_message = str(e)
                print(f"Caught an exception: {exception_message}")
    def sendMessage(self,text):
        if(sendtotelegram):
            try:
                self.Message = self.telegrambot.sendMessage(chat_id, text)
            except Exception as e:
                exception_message = str(e)
                print(f"Caught an exception: {exception_message}")
    
    def Progress(self,iter,NumberofEpisodes):
            if(sendtotelegram):
                if (iter % round(NumberofEpisodes/50)==0):
                    try:
                        self.telegrambot.editMessageText((chat_id, self.running_message['message_id']),'~Running:  '+str(100*iter/NumberofEpisodes)+'% ~~~~~~')
                    except:
                        pass
    def send_photo(self,address,caption):
        if (sendtotelegram):
            self.telegrambot.sendPhoto(chat_id, photo=open(address, 'rb'), caption=caption)
    def send_document(self,address,caption):
        if (sendtotelegram):
            self.telegrambot.sendDocument(chat_id, document=open(address, "rb"), caption=caption)
            
    def sendresults(self,parameters_text,text,M,optimal_values,test_finish_times,Number_of_runs,filename_png,result_text,filename):
        if (sendtotelegram):
            self.telegrambot.sendMessage(chat_id,text)
            for m in range(M):
                self.telegrambot.sendPhoto(chat_id, photo=open(filename_png+f'/DAG_{m}.png', 'rb'), caption=f'DAG of user {m}')
                self.telegrambot.sendPhoto(chat_id, 
                                    photo=open(filename_png+f'/DAG_sol{m}.png', 'rb'), 
                                    caption=f'Optimization solution of user {m} - Optimal Value: {optimal_values[0][0][m]}')
                self.telegrambot.sendPhoto(chat_id, photo=open(filename_png+f'/DAG_{m}_RL.png', 'rb'), caption=f'ML solution of user {m} - Finish Time: {np.mean([test_finish_times[j][m] for j in range(Number_of_runs)])}')
                self.telegrambot.sendPhoto(chat_id, photo=open(filename_png + f'/average_{m}.png', 'rb'), caption=result_text)
            self.telegrambot.sendPhoto(chat_id, photo=open(filename_png + '/moving_averages.png', 'rb'), caption=result_text)
            self.telegrambot.sendPhoto(chat_id, photo=open(filename_png+'/learning_model.png', 'rb'), caption=result_text)
            self.telegrambot.sendDocument(chat_id, document=open(filename+"_Q_result.txt", "rb"),caption='prediction results')
            self.telegrambot.sendDocument(chat_id, document=open(filename + ".txt", "rb"))
            self.telegrambot.deleteMessage((chat_id, self.running_message['message_id']))
        
        self.generate_results_markdown(parameters_text,text, M, optimal_values, test_finish_times, Number_of_runs, filename_png, result_text, filename)

        #telegrambot.deleteMessage((chat_id, Message['message_id']))
    #except:
    #    print('No network')

    # Function to get the chat_id
    def get_chat_id(self,msg):
        content_type, chat_type, chat_id = telepot.glance(msg)
        return chat_id

    # Function to handle incoming messages
    def on_message(self,msg):
        chat_id = self.get_chat_id(msg)
        self.telegrambot.sendMessage(chat_id, f'Your chat ID is: {chat_id}')


    def generate_results_html(self, parameters_text, text, M, optimal_values, test_finish_times, Number_of_runs, filename_png, result_text, filename):
        # Copy the PNG files to a new directory
        source_directory = filename_png
        destination_directory = 'html/'+filename+'/contents'
        os.makedirs(destination_directory, exist_ok=True)
        png_files = [f'DAG_{m}.png' for m in range(M)]
        for m in range(M):
            png_files.append(f'DAG_sol{m}.png')
            png_files.append(f'DAG_{m}_RL.png')

        png_files.append('learning_model.png')
        for png_file in png_files:
            source_path = os.path.join(source_directory, png_file)
            destination_path = os.path.join(destination_directory, png_file)
            shutil.copy2(source_path, destination_path)
        
        source_path = [f'{filename_png}/average_{m}.png' for m in range(M)]
        destination_path = [f'html/{filename}/contents/average_{m}.png' for m in range(M)]
        for s,d in zip(source_path,destination_path):
            shutil.copy2(s, d)
        shutil.copy2(filename_png + '/moving_averages.png', f'html/{filename}/contents/moving_averages.png')
        # Construct the HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Results</title>
        </head>
        <body>
            <pre>{parameters_text}</pre>"
            <h1>{text}</h1>
            <h2>Optimal Values:</h2>
        """
        # Add optimal values to the HTML content
        
        for m in range(M):
            html_content += f"<p>Optimization solution of user {m} - Optimal Value: {optimal_values[0][0][m]}</p>"

        # Add test finish times to the HTML content
        html_content += "<h2>ML Solution Finish Times:</h2>"
        for m in range(M):
            mean_finish_time = np.mean([test_finish_times[j][m] for j in range(Number_of_runs)])
            html_content += f"<p>ML solution of user {m} - Finish Time: {mean_finish_time}</p>"

        # Add image tags for the copied PNG files
        html_content += "<h2>Result Images:</h2>"
        for m in range(M):
            html_content += f"<h2>User{m}:</h2>"
            html_content += f"<img src='contents/average_{m}.png' alt='Average {m}'>"
            html_content += f"<img src='contents/DAG_{m}.png' alt='DAG of user {m}'>"
            html_content += f"<img src='contents/DAG_sol{m}.png' alt='Optimization solution of user {m} - Optimal Value: {optimal_values[0][0][m]}'>"
            html_content += f"<img src='contents/DAG_{m}_RL.png' alt='ML solution of user {m} - Finish Time: {np.mean([test_finish_times[j][m] for j in range(Number_of_runs)])}'>"
        html_content += f"<h2>Average:</h2>"
        html_content += f"<img src='contents/moving_averages.png' alt='Moving Averages'>"
        html_content += "<img src='contents/learning_model.png' alt='Learning Model'>"

        # Close the HTML page
        html_content += """
        </body>
        </html>
        """

        # Save the HTML content to a file
        with open(f'html/{filename}/index.html', 'w') as html_file:
            html_file.write(html_content)


    def generate_results_markdown(self, parameters_text, text, M, optimal_values, test_finish_times, Number_of_runs, filename_png, result_text, filename):
        # Copy the PNG files to a new directory
        source_directory = filename_png
        destination_directory = f'markdown/{filename}/contents'
        os.makedirs(destination_directory, exist_ok=True)
        png_files = [f'DAG_{m}.png' for m in range(M)] + ['learning_model.png']
        
        for m in range(M):
            png_files.append(f'DAG_sol{m}.png')
            png_files.append(f'DAG_{m}_RL.png')

        for png_file in png_files:
            source_path = os.path.join(source_directory, png_file)
            destination_path = os.path.join(destination_directory, png_file)
            shutil.copy2(source_path, destination_path)

        source_path = [f'{filename_png}/average_{m}.png' for m in range(M)]
        destination_path = [f'markdown/{filename}/contents/average_{m}.png' for m in range(M)]
        for s, d in zip(source_path, destination_path):
            shutil.copy2(s, d)
        shutil.copy2(filename_png + '/moving_averages.png', f'markdown/{filename}/contents/moving_averages.png')

        # Construct the Markdown content
        markdown_content = f"# {text}\n\n```\n{parameters_text}\n```\n\n## Optimal Values:\n\n"
        
        # Add optimal values to the Markdown content
        for m in range(M):
            markdown_content += f"- Optimization solution of user {m} - Optimal Value: {optimal_values[0][0][m]}\n"

        # Add test finish times to the Markdown content
        markdown_content += "\n## ML Solution Finish Times:\n\n"
        for m in range(M):
            mean_finish_time = np.mean([test_finish_times[j][m] for j in range(Number_of_runs)])
            markdown_content += f"- ML solution of user {m} - Finish Time: {mean_finish_time}\n"

        # Add image links for the copied PNG files
        markdown_content += "\n## Result Images:\n\n"
        for m in range(M):
            markdown_content += f"### User {m}:\n"
            markdown_content += f"![Average {m}](contents/average_{m}.png)\n"
            markdown_content += f"![DAG of user {m}](contents/DAG_{m}.png)\n"
            markdown_content += f"![Optimization solution of user {m} - Optimal Value: {optimal_values[0][0][m]}](contents/DAG_sol{m}.png)\n"
            markdown_content += f"![ML solution of user {m} - Finish Time: {np.mean([test_finish_times[j][m] for j in range(Number_of_runs)])}](contents/DAG_{m}_RL.png)\n"

        markdown_content += "### Average:\n"
        markdown_content += "![Moving Averages](contents/moving_averages.png)\n"
        markdown_content += "![Learning Model](contents/learning_model.png)\n"

        # Save the Markdown content to a file
        with open(f'markdown/{filename}/README.md', 'w') as markdown_file:
            markdown_file.write(markdown_content)

# Example usage
# generate_results_markdown(self, parameters_text, text, M, optimal_values, test_finish_times, Number_of_runs, filename_png, result_text, filename)

# Define your Telegram class and sendtotelegram variable here

# Define your text_content, M, optimal_values, test_finish_times, Number_of_runs, filename_png, result_text, and filename


