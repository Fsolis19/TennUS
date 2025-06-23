from locust import HttpUser, task, between

class TennusUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def upload_video(self):
        with open("video.mp4", "rb") as f:
            print("Archivo cargado correctamente:", f.name)
            files = {
                "video": ("video.mp4", f, "video/mp4") 
            }
            response = self.client.post("/process", files=files)
            response.raise_for_status()

