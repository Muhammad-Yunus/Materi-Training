wget https://raw.githubusercontent.com/Muhammad-Yunus/Materi-Training/main/A.%20Raspberry%20Pi%20-%20Image%20Processing/%231%20-%20Installation%20%26%20Setup/jupyter.service
wget https://raw.githubusercontent.com/Muhammad-Yunus/Materi-Training/main/A.%20Raspberry%20Pi%20-%20Image%20Processing/%231%20-%20Installation%20%26%20Setup/run_notebook.sh

sudo chmod +x run_notebook.sh
sudo mv jupyter.service /etc/systemd/system/jupyter.service

sudo systemctl enable /etc/systemd/system/jupyter.service
sudo systemctl daemon-reload
sudo systemctl start jupyter.service
sudo systemctl status jupyter.service

