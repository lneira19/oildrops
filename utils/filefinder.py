import os
base_dir = open("basepath.txt", "r").read()

def getFilesFromDirectory(directory, extensions=None):
    """
    Retrieves files from a specified directory with optional filtering by file extensions.
    :param directory: The directory path to search for files.
    :param extensions: A list of file extensions to filter by (e.g., ['.jpg', '.png']). If None, all files are returned."""
    
    complete_directory = base_dir + directory
    
    if not os.path.isdir(complete_directory):
        raise ValueError(f"The provided path '{complete_directory}' is not a valid directory.")
    
    list_files = []
    list_filenames = []

    for root, _, filenames in os.walk(complete_directory):
        for filename in filenames:
            if extensions is None or any(filename.endswith(ext) for ext in extensions):
                list_files.append(os.path.join(root, filename))
                list_filenames.append(filename)
    
    # Sort the lists based on filenames
    list_files.sort()
    list_filenames.sort()

    return list_files, list_filenames

def main():
    # Example usage
    directory = "Gotas_01/"
    extensions = ".jpg"
    
    list_files, list_filenames = getFilesFromDirectory(directory, extensions)

    for file, filename in zip(list_files, list_filenames):
        print(f"File: {file}, Filename: {filename}")

if __name__ == "__main__":
    main()