from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
from werkzeug.security import generate_password_hash, check_password_hash
from functions import img_predict, get_diseases_classes, get_crop_recommendation, get_fertilizer_recommendation, soil_types, Crop_types, crop_list
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Model for User
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)  # New admin flag

# 2. Create a function to add an admin user if one doesn't exist
def add_admin_column_if_not_exists():
    with app.app_context():
        # Check if the column exists
        from sqlalchemy import inspect, text
        inspector = inspect(db.engine)
        columns = [col['name'] for col in inspector.get_columns('user')]
        
        if 'is_admin' not in columns:
            # Add the column
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE user ADD COLUMN is_admin BOOLEAN DEFAULT FALSE'))
                conn.commit()
            print("Added is_admin column to user table")  
              
def create_admin_if_not_exists():
    with app.app_context():
        try:
            admin_user = User.query.filter_by(email='admin@farmersassistant.com').first()
            if not admin_user:
                admin_password = generate_password_hash('admin123', method='pbkdf2:sha256')
                admin = User(name='Admin', email='admin@farmersassistant.com', 
                            password=admin_password, is_admin=True)
                db.session.add(admin)
                db.session.commit()
                print("Admin user created successfully!")
            else:
                # Update existing admin user to ensure it has admin privileges
                if not getattr(admin_user, 'is_admin', False):
                    admin_user.is_admin = True
                    db.session.commit()
                    print("Updated existing user to have admin privileges")
                else:
                    print("Admin user already exists")
        except Exception as e:
            print(f"Error creating/updating admin user: {e}")
            db.session.rollback()

# 3. Call this function after db.create_all()
with app.app_context():
    db.create_all()
    add_admin_column_if_not_exists()  # Add this line
    create_admin_if_not_exists()

# ------------------ Authentication Routes ------------------

@app.route('/')
def home():
    if 'user_id' in session:
        return render_template('index.html', user=session['user_name'])
    return redirect(url_for('login'))

# 4. Modify login route to check for admin status
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['user_name'] = user.name
            session['is_admin'] = user.is_admin
            flash('Login successful!', 'success')
            
            # Redirect admin to admin dashboard
            if user.is_admin:
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('home'))
        else:
            flash('Invalid email or password', 'danger')

    return render_template('login.html')

# 5. Add admin dashboard route
@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('is_admin', False):
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('home'))
    
    # Get all users
    users = User.query.all()
    return render_template('admin_dashboard.html', users=users)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']  # Changed 'username' to 'name'
        email = request.form['email']  # Added email
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        if User.query.filter_by(email=email).first():  # Check by email instead of username
            flash('Email already registered. Please login.', 'warning')
            return redirect(url_for('login'))

        new_user = User(name=name, email=email, password=hashed_password)  # Store name & email
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

# ------------------ Main App Routes ------------------

@app.route('/crop-recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = get_crop_recommendation(to_predict_list)
        return render_template("recommend_result.html", result=result)  # Pass full details
    else:
        return render_template('crop-recommend.html')



@app.route('/fertilizer-recommendation', methods=['GET', 'POST'])
def fertilizer_recommendation():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == "POST":
        to_predict_list = list(map(float, request.form.values()))
        result = get_fertilizer_recommendation(
            num_features=to_predict_list[:-2],
            cat_features=to_predict_list[-2:]
        )
        return render_template("recommend_result.html", result=result)

    return render_template('fertilizer-recommend.html', soil_types=enumerate(soil_types), crop_types=enumerate(Crop_types))

@app.route('/crop-disease', methods=['GET', 'POST'])
def find_crop_disease():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == "POST":
        file = request.files["file"]
        crop = request.form["crop"]
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        prediction = img_predict(file_path, crop)
        disease_name, disease_data = get_diseases_classes(crop, prediction)  # Now returns both name and data
        return render_template('disease-prediction-result.html', 
                               image_file_name=file.filename, 
                               result=disease_name,
                               disease_info=disease_data)  # Pass disease_data to template

    return render_template('crop-disease.html', crops=crop_list)
from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(debug=True)
