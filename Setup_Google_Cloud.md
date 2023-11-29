Google Cloud Vision with Python

1. **Create a Gmail Account**: If you haven't already, sign up for a Gmail account through the google website (<https://www.google.com/>).
1. **Open the Google Cloud Console:** Navigate to <https://www.console.cloud.google.com>.
   1. Log into your account
   1. Click the bar at the top which says “My First Project”, then at the top right of the popup window, click “New Project”
   1. Establish a project name. For mine, I chose CloudOCR. Click create.
1. **Enable the Vision API Service:** Open the menu on the left side of the screen, expand to see “more products”, go down to APIs & Services, and click on “Library”
   1. Next, search for “Cloud Vision API”, open it, and click “enable”. Then click “Create Credentials”
      1. If this is already enabled, click “manage”. 
   1. Once this opens, on the left side, click “Credentials”
   1. At the top, click “Create Credentials”, then choose “Service account”
   1. Type in any name under “Service account ID”. I chose ocracct. Click create
   1. Once this is done, go into the service account you just created, go to keys, and click “add key”. Keep the default JSON, and click create. This will create a file and saves it to your computer.
   1. Take this new file and name it something recognizable, save it in your working directory (i.e. where the Jupyter Notebook file is).
      1. I saved mine as “ServiceAccountToken.json”
      1. My working directory is “/Users/michaelgiordano/Documents/GitHub/OCR\_Improvements/OCR\_Python”
