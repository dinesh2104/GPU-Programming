import http.client
import os
import mimetypes
from io import BytesIO
import argparse

def submit_data(roll_no, tc, int1, int2, file_path):
    # Define the server and port
    server = "10.24.6.48"  # Replace <server_ip> with the actual server IP address
    port = 8000

    # Check if the .cu file exists
    if not os.path.isfile(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return

    # Prepare the form data (roll_no, int1, int2)
    data = {
        'roll_no': roll_no,
        'tc': str(tc),
        'int1': str(int1),
        'int2': str(int2)
    }

    # Prepare the file to be uploaded
    file_name = f"{roll_no}.cu"
    file_type, _ = mimetypes.guess_type(file_path)
    if file_type is None:
        file_type = 'application/octet-stream'

    # Read the content of the .cu file
    with open(file_path, 'rb') as file:
        file_content = file.read()

    # Build multipart/form-data body
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    body = BytesIO()

    # Add form data fields
    for key, value in data.items():
        body.write(f'--{boundary}\r\n'.encode())
        body.write(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        body.write(f'{value}\r\n'.encode())

    # Add file data
    body.write(f'--{boundary}\r\n'.encode())
    body.write(f'Content-Disposition: form-data; name="file"; filename="{file_name}"\r\n'.encode())
    body.write(f'Content-Type: {file_type}\r\n\r\n'.encode())
    body.write(file_content)
    body.write(f'\r\n--{boundary}--\r\n'.encode())

    # Set the headers
    headers = {
        'Content-Type': f'multipart/form-data; boundary={boundary}',
        'Content-Length': str(len(body.getvalue()))
    }

    # Establish connection
    conn = http.client.HTTPConnection(server, port)
    
    # Send the POST request
    conn.request('POST', '/submit', body=body.getvalue(), headers=headers)

    # Get the response
    response = conn.getresponse()
    if response.status == 200:
        response_data = response.read().decode('utf-8')
        print(f"Submission successful! Response: {response_data}")
    else:
        print(f"Error: {response.status} {response.reason}")

    # Close the connection
    conn.close()

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Submit student data and a .cu file.")
    parser.add_argument('roll_no', help="Roll number of the student")
    parser.add_argument('tc', help="Testacse number 1 or 2")
    parser.add_argument('int1', type=int, help="First integer (int1)")
    parser.add_argument('int2', type=int, help="Second integer (int2)")
    parser.add_argument('file_path', help="Path to the .cu file")

    # Parse arguments
    args = parser.parse_args()

    # Call the function to submit the data
    submit_data(args.roll_no, args.tc, args.int1, args.int2, args.file_path)
