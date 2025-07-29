"""
File handling utilities for YOLO web application
"""

import os
import uuid
import shutil
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import mimetypes

class FileHandler:
    """Handle file operations for uploads and results"""
    
    def __init__(self, upload_folder="uploads", results_folder="results", 
                 allowed_extensions=None, max_file_size=16*1024*1024):
        self.upload_folder = upload_folder
        self.results_folder = results_folder
        self.max_file_size = max_file_size
        
        if allowed_extensions is None:
            self.allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
        else:
            self.allowed_extensions = allowed_extensions
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.upload_folder, exist_ok=True)
        os.makedirs(self.results_folder, exist_ok=True)
    
    def is_allowed_file(self, filename):
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def save_upload(self, file):
        """
        Save uploaded file with unique name
        
        Args:
            file: FileStorage object from Flask request
            
        Returns:
            dict: Result with success status and file info
        """
        try:
            # Validate file
            if not file or file.filename == '':
                return {'success': False, 'error': 'No file provided'}
            
            if not self.is_allowed_file(file.filename):
                return {'success': False, 'error': 'File type not allowed'}
            
            # Check file size (if possible)
            file.seek(0, 2)  # Seek to end
            size = file.tell()
            file.seek(0)     # Reset to beginning
            
            if size > self.max_file_size:
                return {'success': False, 'error': 'File too large'}
            
            # Generate unique filename
            original_filename = secure_filename(file.filename)
            file_extension = original_filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
            
            # Save file
            file_path = os.path.join(self.upload_folder, unique_filename)
            file.save(file_path)
            
            return {
                'success': True,
                'filename': unique_filename,
                'original_filename': original_filename,
                'file_path': file_path,
                'size': size
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Failed to save file: {str(e)}'}
    
    def save_result_image(self, image_array, original_filename):
        """
        Save result image
        
        Args:
            image_array: OpenCV image array
            original_filename: Original upload filename for reference
            
        Returns:
            dict: Result with success status and file info
        """
        try:
            import cv2
            
            # Generate result filename
            base_name = os.path.splitext(original_filename)[0]
            result_filename = f"result_{base_name}_{uuid.uuid4().hex[:8]}.jpg"
            result_path = os.path.join(self.results_folder, result_filename)
            
            # Save image
            cv2.imwrite(result_path, image_array)
            
            return {
                'success': True,
                'filename': result_filename,
                'file_path': result_path
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Failed to save result: {str(e)}'}
    
    def get_file_path(self, filename, folder_type='upload'):
        """Get full path to file"""
        if folder_type == 'upload':
            return os.path.join(self.upload_folder, filename)
        elif folder_type == 'result':
            return os.path.join(self.results_folder, filename)
        else:
            raise ValueError("folder_type must be 'upload' or 'result'")
    
    def delete_file(self, filename, folder_type='upload'):
        """Delete file"""
        try:
            file_path = self.get_file_path(filename, folder_type)
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception:
            return False
    
    def cleanup_old_files(self, max_age_hours=24):
        """
        Clean up files older than specified hours
        
        Args:
            max_age_hours (int): Maximum age of files in hours
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        for folder in [self.upload_folder, self.results_folder]:
            try:
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path):
                        file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                        if file_time < cutoff_time:
                            os.remove(file_path)
            except Exception as e:
                print(f"Error cleaning up {folder}: {e}")
    
    def get_file_info(self, filename, folder_type='upload'):
        """Get information about a file"""
        try:
            file_path = self.get_file_path(filename, folder_type)
            
            if not os.path.exists(file_path):
                return None
            
            stat = os.stat(file_path)
            mime_type, _ = mimetypes.guess_type(file_path)
            
            return {
                'filename': filename,
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'mime_type': mime_type
            }
            
        except Exception:
            return None
    
    def list_files(self, folder_type='upload'):
        """List all files in specified folder"""
        try:
            if folder_type == 'upload':
                folder = self.upload_folder
            elif folder_type == 'result':
                folder = self.results_folder
            else:
                return []
            
            files = []
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    info = self.get_file_info(filename, folder_type)
                    if info:
                        files.append(info)
            
            return sorted(files, key=lambda x: x['created'], reverse=True)
            
        except Exception:
            return []
